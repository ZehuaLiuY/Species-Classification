import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    gamma, alpha adjustable，normally gamma=2, alpha=1。
    """
    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes], logits not softmax
        targets: [batch_size], real class index (not one-hot)
        """
        log_pt = F.log_softmax(inputs, dim=1)
        pt = log_pt.exp()  # softmax of logits

        log_pt = log_pt.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        loss = - self.alpha * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# criterion = FocalLoss(gamma=2.0, alpha=1.0, reduction='mean').cuda()

class LDAMLoss(nn.Module):
    """
        Large Margin Distance Loss
        1. calculate the margin for each class
        2. subtract the margin from the logits
        3. apply softmaz
    """

    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None, reduction='mean'):
        super(LDAMLoss, self).__init__()
        self.cls_num_list = cls_num_list
        self.max_m = max_m
        self.s = s
        self.weight = weight
        self.reduction = reduction

        m_list = []
        base = min(self.cls_num_list) ** 0.25
        for freq in self.cls_num_list:
            m_list.append(self.max_m * (1 - freq**0.25 / base))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1,1), 1)

        self.m_list = self.m_list.to(x.device)

        batch_m = []
        for i in range(len(target)):
            cls_id = target[i]
            batch_m.append(self.m_list[cls_id])
        batch_m = torch.stack(batch_m)

        x_m = x.clone()
        x_m[index.bool()] -= batch_m

        output = self.s * x_m

        return F.cross_entropy(output, target, weight=self.weight, reduction=self.reduction)

# cls_num_list = [cls_counts[i] for i in range(NUM_CLASSES)]
# cls_num_list = np.array(cls_num_list)
#
# weight_t = None
#
# criterion = LDAMLoss(
#     cls_num_list=cls_num_list,
#     max_m=0.5,   # 可调
#     s=30,       # 可调
#     weight=weight_t,
#     reduction='mean'
# ).cuda()


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, reduction='mean'):
        """
        cls_num_list: list of int, number of samples for each class
        """
        super(BalancedSoftmaxLoss, self).__init__()
        self.cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, logits, label):
        """
        logits: [batch_size, num_classes], 未过softmax
        label:  [batch_size]
        """
        cls_num_list = self.cls_num_list.to(logits.device)

        # Balanced Softmax: log_prob = x_ij + log(n_j) - log(sum_i(n_i * exp(x_i)))
        batch_size, num_classes = logits.shape
        label_expand = label.unsqueeze(1)  # [batch_size, 1]

        log_n = torch.log(cls_num_list)  # shape=[num_classes]
        log_n = log_n.unsqueeze(0).expand(batch_size, num_classes)  # [B, C]


        logits_adjusted = logits + log_n

        log_prob = F.log_softmax(logits_adjusted, dim=1)

        loss = F.nll_loss(log_prob, label, reduction=self.reduction)
        return loss


# cls_num_list = [cls_counts[i] for i in range(NUM_CLASSES)]
#
# criterion = BalancedSoftmaxLoss(cls_num_list).cuda()
#