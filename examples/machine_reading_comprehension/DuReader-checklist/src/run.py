# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time

import numpy as np
import paddle
import glob
import re
import warnings

from paddle.io import DataLoader
from args import parse_args
import json

from paddlenlp.metrics.squad import squad_evaluate
from squad import DuReaderChecklist, compute_prediction_checklist
from paddlenlp.metrics.squad import compute_prediction
from paddlenlp.data import Pad, Stack, Dict
from adv import FGM, PGD
from models import ErnieForQuestionAnswering, BertForQuestionAnswering, RobertaForQuestionAnswering
from paddlenlp.transformers import BertTokenizer, ErnieTokenizer, RobertaTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.datasets import load_dataset
import paddle.nn.functional as F

warnings.filterwarnings("ignore")
from dice_loss import DiceLoss

MODEL_CLASSES = {
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaForQuestionAnswering, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def make_answer_content(inputs, start_positions, end_positions):
    shape = inputs.shape[0:2]
    answer_content_labels = paddle.zeros(shape, dtype="int64")
    for i, (start_position,
            end_position) in enumerate(zip(start_positions, end_positions)):
        answer_content_labels[i][start_position:end_position] = paddle.ones(
            (end_position - start_position), dtype="int64")
    return answer_content_labels


def create_masked_lm_predictions(tokens, masked_lm_prob, tokenizer, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == tokenizer.cls_token_id or token == tokenizer.sep_token_id:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(20, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = tokenizer.mask_token_id
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = rng.randint(5, len(tokenizer.vocab) - 1)

        output_tokens[index] = masked_token

        masked_lms.append((index, tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x[0])

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])

    return (output_tokens, masked_lm_positions, masked_lm_labels)


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits, cls_logits, content_logits = y
        start_position, end_position, answerable_label, content_label = label

        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        answerable_label = paddle.unsqueeze(answerable_label, axis=-1)
        content_label = paddle.unsqueeze(content_label, axis=-1)
        start_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=start_logits, label=start_position, soft_label=False)
        start_loss = paddle.mean(start_loss)
        end_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=end_logits, label=end_position, soft_label=False)
        end_loss = paddle.mean(end_loss)
        cls_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=cls_logits, label=answerable_label, soft_label=False)
        cls_loss = paddle.mean(cls_loss)
        content_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=content_logits, label=content_label, soft_label=False)
        content_loss = paddle.mean(content_loss)
        loss = (cls_loss * 0.33 + start_loss * 0.33 + end_loss * 0.33)
        return loss


class CrossEntropyLossForChecklist(paddle.nn.Layer):
    def __init__(self, dice_loss):
        super(CrossEntropyLossForChecklist, self).__init__()
        self.dice_loss = dice_loss

    def forward(self, y, label):
        start_logits, end_logits, cls_logits, content_logits = y
        batch_size, ignored_index = start_logits.shape
        start_position, end_position, answerable_label = label

        cls_logits = paddle.reshape(cls_logits, [-1, 1])

        start_labels = F.one_hot(start_position, num_classes=ignored_index)
        end_labels = F.one_hot(end_position, num_classes=ignored_index)
        cls_labels = F.one_hot(answerable_label, num_classes=2)

        cls_labels = paddle.reshape(cls_labels, [-1, 1])

        start_loss = self.dice_loss(start_logits, start_labels)
        end_loss = self.dice_loss(end_logits, end_labels)

        cls_loss = self.dice_loss(cls_logits, cls_labels)

        loss = cls_loss * 0.8 + start_loss * 0.1 + end_loss * 0.1
        return loss


@paddle.no_grad()
def evaluate(model, data_loader, args, prefix=""):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    all_cls_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor, cls_logits_tensor, _ = model(
            input_ids=input_ids, token_type_ids=token_type_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])
            all_cls_logits.append(cls_logits_tensor.numpy()[idx])

    all_predictions, all_nbest_json, all_cls_predictions = compute_prediction_checklist(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits, all_cls_logits), True,
        args.n_best_size, args.max_answer_length, args.cls_threshold)
    '''
    all_predictions, all_nbest_json, all_cls_predictions = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), args.version_2_with_negative,
        args.n_best_size, args.max_answer_length,
        0)
    '''
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(
            os.path.join(args.output_dir, prefix + '_predictions.json'),
            "w",
            encoding='utf-8') as writer:
        writer.write(
            json.dumps(
                all_predictions, ensure_ascii=False, indent=4) + "\n")

    with open(
            os.path.join(args.output_dir, prefix + '_nbest_predictions.json'),
            "w",
            encoding="utf8") as writer:
        writer.write(
            json.dumps(
                all_nbest_json, indent=4, ensure_ascii=False) + u"\n")
    '''
    if all_cls_predictions:
        with open(os.path.join(args.output_dir,prefix+"_cls_preditions.json"), "w") as f_cls:
            for cls_predictions in all_cls_predictions:
                qas_id, pred_cls_label, no_answer_prob, answerable_prob = cls_predictions
                f_cls.write('{}\t{}\t{}\t{}\n'.format(qas_id, pred_cls_label, no_answer_prob, answerable_prob))
    '''
    model.train()


def run(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    rng = random.Random()
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    set_seed(args)

    if paddle.distributed.get_rank() == 0:
        if os.path.exists(args.model_name_or_path):
            print("init checkpoint from %s" % args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        contexts = [examples[i]['context'] for i in range(len(examples))]
        questions = [examples[i]['question'] for i in range(len(examples))]

        tokenized_examples = tokenizer(
            questions,
            contexts,
            stride=args.doc_stride,
            max_seq_len=args.max_seq_length)

        for i, tokenized_example in enumerate(tokenized_examples):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_example["input_ids"]

            masked_input, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                input_ids, 0.15, tokenizer, rng)

            tokenized_examples[i]["masked_input"] = masked_input
            tokenized_examples[i]["masked_lm_positions"] = masked_lm_positions
            tokenized_examples[i]["masked_lm_labels"] = masked_lm_labels
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offsets = tokenized_example['offset_mapping']

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_example['token_type_ids']

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = tokenized_example['overflow_to_sample']
            answers = examples[sample_index]['answers']
            answer_starts = examples[sample_index]['answer_starts']
            is_impossible = examples[sample_index]['is_impossible']

            tokenized_examples[i]['label_mask'] = [1] + tokenized_example[
                'token_type_ids'][1:]

            # If no answers are given, set the cls_index as answer.
            if is_impossible:
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
                tokenized_examples[i]['answerable_label'] = 0
            else:
                # Start/end character index of the answer in the text.
                start_char = answer_starts[0]
                end_char = start_char + len(answers[0])
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 2
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and
                        offsets[token_end_index][1] >= end_char):
                    tokenized_examples[i]["start_positions"] = cls_index
                    tokenized_examples[i]["end_positions"] = cls_index
                    tokenized_examples[i]['answerable_label'] = 0
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[
                            token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples[i][
                        "start_positions"] = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples[i]["end_positions"] = token_end_index + 1
                    tokenized_examples[i]['answerable_label'] = 1

        return tokenized_examples

    if args.do_train:
        assert args.train_file != None, "--train_file should be set when training!"
        train_ds = DuReaderChecklist().read(args.train_file)
        dureader_ds = load_dataset(
            'dureader_robust', data_files='src/train_aug.json')
        #train_ds.new_data += dureader_ds.new_data
        #train_ds.new_data += dureader_ds.new_data[:1500]
        #train_ds.new_data += dureader_ds.new_data[-1500:]
        '''
        def check_len(example):
            if len(example['context']) > 128:
                return False
            else:
                return True
        train_ds.filter(check_len,num_workers=8)
        '''
        train_ds.new_data = train_ds.new_data[:2945]
        print(len(train_ds))
        print(train_ds[543])

        def foo(examples):

            return True

        train_ds.filter(foo, num_workers=7)
        print(len(train_ds))
        print(train_ds[543])
        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)

        def _collate_data(data, stack_fn=Stack(dtype="int64")):
            num_fields = 8

            out = [None] * num_fields
            # input_ids, segment_ids, input_mask, masked_lm_positions,
            # masked_lm_labels, next_sentence_labels, mask_token_num
            for i, j in zip(['input_ids', "masked_input"], [0, 5]):
                out[j] = Pad(axis=0, pad_val=tokenizer.pad_token_id)(
                    [x[i] for x in data])
            for i, j in zip(
                ['start_positions', "end_positions", "answerable_label"],
                [2, 3, 4]):
                out[j] = stack_fn([x[i] for x in data])
            out[1] = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)(
                [x['token_type_ids'] for x in data])
            batch_size, seq_length = out[0].shape
            size = num_mask = sum(len(x['masked_lm_positions']) for x in data)
            # Padding for divisibility by 8 for fp16 or int8 usage

            # masked_lm_positions
            # Organize as a 1D tensor for gather or use gather_nd
            out[6] = np.full(size, 0, dtype=np.int32)
            # masked_lm_labels
            out[7] = np.full([size, 1], -1, dtype=np.int64)
            mask_token_num = 0
            for i, x in enumerate(data):
                for j, pos in enumerate(x['masked_lm_positions']):
                    out[6][mask_token_num] = i * seq_length + pos
                    out[7][mask_token_num] = x['masked_lm_labels'][j]
                    mask_token_num += 1
            # mask_token_num
            out.append(np.asarray([mask_token_num], dtype=np.float32))
            return out

        train_data_loader = DataLoader(
            dataset=train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=_collate_data,
            return_list=True)

        num_training_steps = args.max_steps if args.max_steps > 0 else len(
            train_data_loader) * args.num_train_epochs

        if paddle.distributed.get_rank() == 0:
            dev_count = paddle.fluid.core.get_cuda_device_count()
            print("Device count: %d" % dev_count)
            print("Num train examples: %d" % len(train_ds))
            print("Max train steps: %d" % num_training_steps)

        lr_scheduler = LinearDecayWithWarmup(
            args.learning_rate, num_training_steps, args.warmup_proportion)

        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)

        loss_fct = DiceLoss(
            with_logits=True,
            smooth=1,
            ohem_ratio=0,
            alpha=0.01,
            square_denominator=True)
        #criterion = CrossEntropyLossForChecklist(loss_fct)
        criterion = CrossEntropyLossForSQuAD()
        pgd = PGD(model, 1, 0.3)
        pgd_k = 3

        task_split = [0] * len(train_data_loader) + [0] * len(train_data_loader)
        random.shuffle(task_split)

        global_step = 0
        tic_train = time.time()
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                input_ids, token_type_ids, start_positions, end_positions, answerable_label, masked_input, masked_lm_positions, masked_lm_labels, mask_token_num = batch
                answer_content_labels = make_answer_content(
                    input_ids, start_positions, end_positions)

                if task_split[(epoch % 2) * len(train_data_loader) + step] == 0:
                    logits = model(
                        input_ids=input_ids, token_type_ids=token_type_ids)
                    loss = criterion(logits,
                                     (start_positions, end_positions,
                                      answerable_label, answer_content_labels))
                    loss.backward()
                else:
                    logits = model(
                        input_ids=masked_input,
                        token_type_ids=token_type_ids,
                        masked_position=masked_lm_positions)
                    masked_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
                        logits, masked_lm_labels, ignore_index=-1)
                    masked_lm_loss = paddle.sum(masked_lm_loss) / mask_token_num
                    masked_lm_loss.backward()

                if global_step % args.logging_steps == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss,
                           args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                '''
                pgd.grad_backup()
                for k in range(pgd_k):
                    pgd.attack(is_first_attack=(k == 0))
                    if k != pgd_k - 1:
                        model.clear_gradients()
                    else:
                        pgd.restore_grad()
                    logits_adv = model(input_ids=input_ids, token_type_ids=token_type_ids)
                    loss_adv = criterion(logits_adv, (start_positions, end_positions,answerable_label))

                    loss_adv.backward()

                pgd.restore()
                '''

                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % args.save_steps == 0 or global_step == num_training_steps:
                    if paddle.distributed.get_rank() == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        print('Saving checkpoint to:', output_dir)

    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        contexts = [examples[i]['context'] for i in range(len(examples))]
        questions = [examples[i]['question'] for i in range(len(examples))]

        tokenized_examples = tokenizer(
            questions,
            contexts,
            stride=args.doc_stride,
            max_seq_len=args.max_seq_length)

        # For validation, there is no need to compute start and end positions
        for i, tokenized_example in enumerate(tokenized_examples):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_example['token_type_ids']

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = tokenized_example['overflow_to_sample']
            tokenized_examples[i]["example_id"] = examples[sample_index]['id']

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples[i]["offset_mapping"] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_example["offset_mapping"])
            ]

        return tokenized_examples

    if args.do_pred:
        input_files = []
        assert args.predict_file != None, "--predict_file should be set when predicting!"
        for input_pattern in args.predict_file:
            input_files.extend(glob.glob(input_pattern))
        assert len(input_files) > 0, 'Can not find predict_file {}'.format(
            args.predict_file)
        for input_file in input_files:
            print('Run prediction on {}'.format(input_file))
            prefix = os.path.basename(input_file)
            prefix = re.sub('.json', '', prefix)
            dev_ds = DuReaderChecklist().read(input_file)
            dev_ds.map(prepare_validation_features, batched=True)

            dev_batch_sampler = paddle.io.BatchSampler(
                dev_ds, batch_size=args.batch_size, shuffle=False)

            dev_batchify_fn = lambda samples, fn=Dict({
                "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
                "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
            }): fn(samples)

            dev_data_loader = DataLoader(
                dataset=dev_ds,
                batch_sampler=dev_batch_sampler,
                collate_fn=dev_batchify_fn,
                return_list=True)
            if paddle.distributed.get_rank() == 0:
                evaluate(
                    model,
                    dev_data_loader,
                    args,
                    prefix=prefix + args.model_type)


if __name__ == "__main__":
    args = parse_args()
    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."

    run(args)
