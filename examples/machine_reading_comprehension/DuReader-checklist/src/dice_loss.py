import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Optional


class DiceLoss(nn.Layer):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)
    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
from src.dice_loss import DiceLoss
import paddle
loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
input = paddle.reshape(paddle.to_tensor([2, 1, 2, 2, 1]).astype('float'),[-1,1])
input.stop_gradient=False
target = paddle.to_tensor([0, 1, 0, 0, 0])
output = loss(input, target)
output.backward()
    """

    def __init__(self,
                 smooth: Optional[float]=1e-4,
                 square_denominator: Optional[bool]=False,
                 with_logits: Optional[bool]=True,
                 ohem_ratio: float=0.0,
                 alpha: float=0.0,
                 reduction: Optional[str]="mean",
                 index_label_position=False) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self, input, target, mask=None):
        logits_size = input.shape[-1]
        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        if self.reduction == "mean":
            return paddle.mean(loss)
        if self.reduction == "sum":
            return paddle.sum(loss)
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input)**self.alpha) * flat_input
        interection = paddle.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) / (
                paddle.sum(flat_input) + paddle.sum(flat_target) + self.smooth))
        else:
            loss = 1 - (
                (2 * interection + self.smooth) /
                (paddle.sum(paddle.square(flat_input, ), -1) + paddle.sum(
                    paddle.square(flat_target), -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = F.one_hot(
            target, num_classes=logits_size).astype(
                'float') if self.index_label_position else target.astype(
                    'float')
        flat_input = paddle.nn.Softmax()(
            flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.astype('float')
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = paddle.ones_like(target)

        loss = None
        if self.ohem_ratio > 0:

            mask_neg = paddle.logical_not(mask.astype('bool'))
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = paddle.sum(pos_example.astype('int'))
                neg_num = paddle.sum(mask.astype('int')) - (
                    pos_num - paddle.sum(
                        paddle.logical_and(mask_neg, pos_example).astype('int'))
                )
                keep_num = min(
                    int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = paddle.reshape(
                        paddle.masked_select(
                            flat_input,
                            paddle.reshape(neg_example,
                                           [-1, 1]).astype('bool')),
                        [-1, logits_size])
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort = paddle.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = paddle.logical_or(
                        (paddle.argmax(flat_input) == label_idx &
                         flat_input[:, label_idx] >= threshold),
                        paddle.reshape(pos_example, [-1]))
                    ohem_mask_idx = cond.astype('int')

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(
                    paddle.reshape(flat_input_idx, [-1, 1]),
                    paddle.reshape(flat_target_idx, [-1, 1]))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(
                    paddle.reshape(flat_input_idx, [-1, 1]),
                    paddle.reshape(flat_target_idx, [-1, 1]))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = paddle.reshape(input, [-1])

        flat_target = paddle.reshape(target, [-1]).astype('float')
        flat_input = paddle.nn.Sigmoid()(
            flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.astype('float')

            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = paddle.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = paddle.sum(pos_example.astype('int')) - paddle.sum(
                paddle.logical_and(pos_example, mask_neg_num).astype('int'))
            neg_num = paddle.sum(neg_example.astype('int'))
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)
            neg_scores = paddle.masked_select(flat_input,
                                              neg_example.astype('bool'))
            neg_scores_sort = paddle.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num + 1]
            cond = paddle.logical_or((flat_input > threshold),
                                     paddle.reshape(pos_example, [-1]))
            ohem_mask = cond.astype('int')
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)
