import torch
import torch.nn as nn
import numpy as np


# Constants
N_PAIR = 'n-pair'
ANGULAR = 'angular'
N_PAIR_ANGULAR = 'n-pair-angular'
MAIN_LOSS_CHOICES = (N_PAIR, ANGULAR, N_PAIR_ANGULAR)

CROSS_ENTROPY = 'cross-entropy'


class BlendedLoss(object):
    def __init__(self, main_loss_type, cross_entropy_flag):
        super(BlendedLoss, self).__init__()
        self.main_loss_type = main_loss_type
        assert main_loss_type in MAIN_LOSS_CHOICES, "invalid main loss: %s" % main_loss_type

        self.metrics = []
        if self.main_loss_type == N_PAIR:
            self.main_loss_fn = NPairLoss()
        elif self.main_loss_type == ANGULAR:
            self.main_loss_fn = AngularLoss()
        elif self.main_loss_type == N_PAIR_ANGULAR:
            self.main_loss_fn = NPairAngularLoss()
        else:
            raise ValueError

        self.cross_entropy_flag = cross_entropy_flag
        self.lambda_blending = 0
        if cross_entropy_flag:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
            self.lambda_blending = 0.3

    def calculate_loss(self, target, output_embedding, output_cross_entropy=None):
        if target is not None:
            target = (target,)

        loss_dict = {}
        blended_loss = 0
        if self.cross_entropy_flag:
            assert output_cross_entropy is not None, "Outputs for cross entropy loss is needed"

            loss_inputs = self._gen_loss_inputs(target, output_cross_entropy)
            cross_entropy_loss = self.cross_entropy_loss_fn(*loss_inputs)
            blended_loss += self.lambda_blending * cross_entropy_loss
            loss_dict[CROSS_ENTROPY + '-loss'] = [cross_entropy_loss.item()]

        loss_inputs = self._gen_loss_inputs(target, output_embedding)
        main_loss_outputs = self.main_loss_fn(*loss_inputs)
        main_loss = main_loss_outputs[0] if type(main_loss_outputs) in (tuple, list) else main_loss_outputs
        blended_loss += (1-self.lambda_blending) * main_loss
        loss_dict[self.main_loss_type+'-loss'] = [main_loss.item()]

        for metric in self.metrics:
            metric(output_embedding, target, main_loss_outputs)

        return blended_loss, loss_dict

    @staticmethod
    def _gen_loss_inputs(target, embedding):
        if type(embedding) not in (tuple, list):
            embedding = (embedding,)
        loss_inputs = embedding
        if target is not None:
            if type(target) not in (tuple, list):
                target = (target,)
            loss_inputs += target
        return loss_inputs


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]


class AngularLoss(NPairLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        super(AngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang
        self.softplus = nn.Softplus()

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.angular_loss(anchors, positives, negatives, self.angle_bound) \
                 + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def angular_loss(anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0]

        x = torch.exp(x - t.unsqueeze(dim=1))
        x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        loss = torch.mean(t + x)

        return loss


class NPairAngularLoss(AngularLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        super(NPairAngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_angular_loss(anchors, positives, negatives, self.angle_bound) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    def n_pair_angular_loss(self, anchors, positives, negatives, angle_bound=1.):
        """
        Calculates N-Pair angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar, n-pair_loss + lambda * angular_loss
        """
        n_pair = self.n_pair_loss(anchors, positives, negatives)
        angular = self.angular_loss(anchors, positives, negatives, angle_bound)

        return (n_pair + self.lambda_ang * angular) / (1+self.lambda_ang)
