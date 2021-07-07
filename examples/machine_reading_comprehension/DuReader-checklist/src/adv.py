import paddle
import paddle.nn.functional as F


class BPP(object):
    def __init__(self, model, beta, mu):
        self.model = model
        self.beta = beta
        self.mu = mu
        self.theta_til = {}
        for name, param in self.model.named_parameters():
            self.theta_til[name] = param.clone()

    def theta_til_backup(self, named_parameters):
        for name, param in named_parameters:
            self.theta_til[name] = (
                1 - self.beta
            ) * param.clone() + self.beta * self.theta_til[name]

    def bregman_divergence(self, batch, logits):

        start_logits, end_logits, cls_logits = logits

        start_theta_prob = F.softmax(start_logits, axis=-1)
        end_theta_prob = F.softmax(end_logits, axis=-1)
        cls_theta_prob = F.softmax(cls_logits, axis=-1)

        param_bak = {}
        for name, param in self.model.named_parameters():
            param_bak[name] = param.clone()
            param.set_value(self.theta_til[name])

        with paddle.no_grad():
            start_logits_til, end_logits_til, cls_logits_til = self.model(
                *batch)

            start_theta_til_prob = F.softmax(start_logits_til, axis=-1)
            end_theta_til_prob = F.softmax(end_logits_til, axis=-1)
            cls_theta_til_prob = F.softmax(cls_logits_til, axis=-1)

        for name, param in self.model.named_parameters():
            param.set_value(param_bak[name])

        start_bregman_divergence = F.kl_div(start_theta_prob.log(), start_theta_til_prob, reduction='batchmean') + \
            F.kl_div(start_theta_til_prob.log(), start_theta_prob, reduction='batchmean')

        end_bregman_divergence = F.kl_div(end_theta_prob.log(), end_theta_til_prob, reduction='batchmean') + \
            F.kl_div(end_theta_til_prob.log(), end_theta_prob, reduction='batchmean')

        cls_bregman_divergence = F.kl_div(cls_theta_prob.log(), cls_theta_til_prob, reduction='batchmean') + \
            F.kl_div(cls_theta_til_prob.log(), cls_theta_prob, reduction='batchmean')

        bregman_divergence = ((start_bregman_divergence + end_bregman_divergence
                               ) / 2 + cls_bregman_divergence) / 2

        return bregman_divergence


class PGD(object):
    def __init__(self, model, epsilon, alpha, eta=1e-3):
        super(PGD, self).__init__()
        self.embed_bak = {}
        self.grad_bak = {}
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.eta = eta

    def attack(self, emb_name='emb', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                if is_first_attack:
                    self.embed_bak[name] = param.clone()
                norm = paddle.norm(param._grad_ivar())
                if norm != 0 and not paddle.isnan(norm):
                    r_at = self.alpha * param._grad_ivar() / norm
                    param.set_value(param + r_at)
                    param.set_value(self.project(name, param))

    def project(self, param_name, param_data):
        r = param_data - self.embed_bak[param_name]
        if paddle.norm(r) > self.epsilon:
            r = self.epsilon * r / paddle.norm(r)
        return self.embed_bak[param_name] + self.eta * r

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.embed_bak
                param.set_value(self.embed_bak[name])
        self.embed_bak = {}

    def grad_backup(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                self.grad_bak[name] = param._grad_ivar().clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.grad_bak
                grad = param._grad_ivar()
                grad[...] = self.grad_bak[name]


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):

        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                self.backup[name] = param.clone()
                norm = paddle.norm(param._grad_ivar())
                if norm != 0 and not paddle.isnan(norm):
                    r_at = epsilon * param._grad_ivar() / norm
                    param.set_value(param + r_at)

    def restore(self, emb_name='emb'):

        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])
        self.backup = {}
