import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# from metrics import dice_coef
# from metrics import dice
from collections import OrderedDict
import warnings
import contextlib
warnings.filterwarnings("ignore")
class DR_loss(torch.nn.Module):
    # for unlabel data
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(DR_loss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        #         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool

    #         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k,arg):
                    # 1,16,8,8  4,16,8,8, arg
        # batch_size = feat_q.shape[0]
        # num_neg = feat_k.shape[0]
        # dim = feat_q.shape[1]
        #         width = feat_q.shape[2]
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        num_leg = feat_k.shape[0]
        feat_q = feat_q.view(batch_size,-1)
        feat_k = feat_k.view(num_leg,-1)
        # 1,16,64  4,16,64, arg
        feat_q = F.normalize(feat_q, dim=-1, p=2)
        feat_k = F.normalize(feat_k, dim=-1, p=2)
        feat_k = feat_k.detach()

        # pos logit
        # 1,16*64  1,16*64, arg
        l_pos = torch.mm(feat_q,feat_k[arg].unsqueeze(0).transpose(1,0))  # 1,1
        feat_k = torch.cat([feat_k[:arg],feat_k[arg+1:]],dim=0)

        # 1,16*64  3,16*64, arg
        l_neg = torch.mm(feat_q, feat_k.transpose(1, 0))  # 1,3


        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  # 1,4

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

class PL_loss(torch.nn.Module):
    # for unlabel data
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(PL_loss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        #         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool
    #         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
    def forward(self, feat_q, feat_k, arg1,arg2):
        # 1,16,8,8  4,16,8,8, arg
        batch_size = feat_q.shape[0]
        num_neg = feat_k.shape[0]

        q_flat = feat_q.view(batch_size,-1)
        k_flat = feat_k.view(num_neg,-1)

        positive_similarities = F.cosine_similarity(q_flat, k_flat[[arg1, arg2]],dim=1)

        negative_indices = [i for i in range(num_neg) if i not in [arg1,arg2]]
        negative_similarities = F.cosine_similarity(q_flat,k_flat[negative_indices],dim=1)
        positive_exp = torch.exp(positive_similarities / self.temperature)
        negative_exp = torch.exp(negative_similarities / self.temperature)
        positive_sum = torch.sum(positive_exp)
        negative_sum = torch.sum(negative_exp)
        positive_loss = -torch.log(positive_sum / (positive_sum + negative_sum))

        return positive_loss

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target,margin=None):
        target = target.float()
        smooth = 1e-5
        if margin != None:
            intersect = torch.sum(score * target * margin)
            y_sum = torch.sum(target * target * margin)
            z_sum = torch.sum(score * score * margin)
        else:
            intersect = torch.sum(score * target)
            y_sum = torch.sum(target * target)
            z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target,margin=None,weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            if margin != None:
                dice = self._dice_loss(inputs[:, i], target[:, i],margin)
            else:
                dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                             keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


###############################################
# BCE = torch.nn.BCELoss()

def weighted_loss(pred, mask):
    BCE = torch.nn.BCELoss(reduction='none')

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask).float()
    wbce = BCE(pred, mask)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def calc_loss(pred, target, bce_weight=0.5):
    bce = weighted_loss(pred, target)
    # dl = 1 - dice_coef(pred, target)
    # loss = bce * bce_weight + dl * bce_weight

    return bce


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    loss1 = calc_loss(logit_S1, labels_S1)
    loss2 = calc_loss(logit_S2, labels_S2)

    return loss1 + loss2


def loss_diff(u_prediction_1, u_prediction_2, batch_size):
    a = weighted_loss(u_prediction_1, Variable(u_prediction_2, requires_grad=False))
    #     print('a',a.size())
    a = a.item()

    b = weighted_loss(u_prediction_2, Variable(u_prediction_1, requires_grad=False))
    b = b.item()

    loss_diff_avg = (a + b)
    #     print('loss_diff_avg',loss_diff_avg)
    #     print('loss_diff batch size',batch_size)
    #     return loss_diff_avg / batch_size
    return loss_diff_avg


###############################################
# contrastive_loss

class ConLoss(torch.nn.Module):
    # for unlabel data
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        #         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool

    #         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        #         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)  # batch * dim * np  # batch * np * dim
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim,
                                                                     1))  # (batch * np) * 1 * dim #(batch * np) * dim * 1  #(batch * np) * 1
        l_pos = l_pos.view(-1, 1)  # (batch * np) * 1

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)  # batch * np * dim
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # batch * np * np

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)  # (batch * np) * np

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  # (batch * np) * (np+1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d
class VAT2d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)

    def forward(self, model, x):
        with torch.no_grad():
            pred= F.softmax(model(x)[0], dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                logp_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)[0]
            logp_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(logp_hat, pred)
        return lds

# class MocoLoss(torch.nn.Module):
# #for unlabel data
#     def __init__(self, temperature=0.07):

#         super(MocoLoss, self).__init__()
#         self.temperature = temperature
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

#     def forward(self, feat_q, feat_k, queue):
#         assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
#         feat_q = F.normalize(feat_q, dim=-1, p=1)
#         feat_k = F.normalize(feat_k, dim=-1, p=1)
#         batch_size = feat_q.shape[0]
#         dim = feat_q.shape[1]
#         K = len(queue)
# #         print('K',K)

#         feat_k = feat_k.detach()

#         # pos logit
#         l_pos = torch.bmm(feat_q.view(batch_size,1,dim),feat_k.view(batch_size,dim,1))  #batch_size * 1
#         l_pos = l_pos.view(-1, 1)
#         feat_k = feat_k.transpose(0,1)
# #         print('feat_k',feat_k.size())
#         # neg logit
#         if K == 0:
#             l_neg = torch.mm(feat_q.view(batch_size,dim), feat_k)
#         else:

#             queue_tensor = torch.cat(queue,dim = 1)
# #             print('queue_tensor.size()',queue_tensor.size())

#             l_neg = torch.mm(feat_q.view(batch_size,dim), queue_tensor) #batch_size * K
# #         print(l_pos.size())
# #         print(l_neg.size())

#         out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #batch_size * (K+1)

# #         print(1)

#         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
#                                                         device=feat_q.device))
# #         print(2)

#         queue.append(feat_k)

#         if K >= 10:
#             queue.pop(0)

#         return loss,queue


class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        #         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool

    #         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        #         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        #         l_pos = torch.zeros((batch_size*2304,1)).cuda()
        #         l_pos = torch.zeros((batch_size*1024,1)).cuda()
        #         l_pos = torch.zeros((batch_size*784,1)).cuda()
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)
        l_pos = torch.zeros((l_neg.size(0), 1)).cuda()
        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


def info_nce_loss(feats1, feats2):
    #     imgs, _ = batch
    #     imgs = torch.cat(imgs, dim=0)

    # Encode all images
    #     feats = self.convnet(imgs)
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats1[:, None, :], feats2[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / 0.07
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    # Logging loss
    #     self.log(mode+'_loss', nll)
    # Get ranking position of positive example
    #     comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
    #                               cos_sim.masked_fill(pos_mask, -9e15)],
    #                              dim=-1)
    #     sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    #     # Logging ranking metrics
    #     self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
    #     self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
    #     self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

    return nll

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)
class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        #         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool

    #         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        #         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))
        l_pos = l_pos.view(-1, 1)
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)
        #         l_pos = torch.zeros((l_neg.size(0),1)).cuda()
        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class MocoLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, use_queue=True, max_queue=1):

        super(MocoLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_queue = use_queue
        self.mask_dtype = torch.bool
        self.queue = OrderedDict()
        self.idx_list = []
        self.max_queue = max_queue

    def forward(self, feat_q, feat_k, idx):
        num_enqueue = 0
        num_update = 0
        num_dequeue = 0
        mid_pop = 0
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        dim = feat_q.shape[1]
        batch_size = feat_q.shape[0]
        feat_q = feat_q.reshape(batch_size, -1)
        feat_k = feat_k.reshape(batch_size, -1)

        K = len(self.queue)
        #         print(K)

        feat_k = feat_k.detach()

        # pos logit
        l_pos = F.cosine_similarity(feat_q, feat_k, dim=1)
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if K == 0 or not self.use_queue:
            l_neg = F.cosine_similarity(feat_q[:, None, :], feat_k[None, :, :], dim=-1)
        else:
            for i in range(0, batch_size):
                if str(idx[i].item()) in self.queue.keys():
                    self.queue.pop(str(idx[i].item()))
                    mid_pop += 1
            queue_tensor = torch.cat(list(self.queue.values()), dim=0)
            l_neg = F.cosine_similarity(feat_q[:, None, :], queue_tensor.reshape(-1, feat_q.size(1))[None, :, :],
                                        dim=-1)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  # batch_size * (K+1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        if self.use_queue:
            for i in range(0, batch_size):
                if str(idx[i].item()) not in self.queue.keys():
                    self.queue[str(idx[i].item())] = feat_k[i].clone()[None, :]
                    num_enqueue += 1
                else:
                    self.queue[str(idx[i].item())] = feat_k[i].clone()[None, :]
                    num_update += 1
                if len(self.queue) >= 1056 + 1:
                    self.queue.popitem(False)

                    num_dequeue += 1

        #         print('queue length, mid pop, enqueue, update queue, dequeue: ', len(self.queue), mid_pop, num_enqueue, num_update, num_dequeue)

        return loss


class ConLoss_queue(torch.nn.Module):
    # for unlabel data
    def __init__(self, temperature=0.07, use_queue=True, max_queue=1):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss_queue, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool
        self.queue = OrderedDict()
        self.idx_list = []
        self.max_queue = max_queue

    def forward(self, feat_q, feat_k):
        num_enqueue = 0
        num_update = 0
        num_dequeue = 0
        mid_pop = 0
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        #         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)  # batch * dim * np  # batch * np * dim
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim,
                                                                     1))  # (batch * np) * 1 * dim #(batch * np) * dim * 1  #(batch * np) * 1
        l_pos = l_pos.view(-1, 1)  # (batch * np) * 1

        # neg logit

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_size, -1, dim)  # batch * np * dim
        feat_k = feat_k.reshape(batch_size, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # batch * np * np

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)  # (batch * np) * np

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  # (batch * np) * (np+1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class MocoLoss_list(torch.nn.Module):
    def __init__(self, temperature=0.07, use_queue=True):

        super(MocoLoss_list, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_queue = use_queue
        self.queue = []
        self.mask_dtype = torch.bool
        self.idx_list = []

    def forward(self, feat_q, feat_k, idx):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        dim = feat_q.shape[1]
        batch_size = feat_q.shape[0]
        feat_q = feat_q.reshape(batch_size, -1)  # 转成向量
        feat_k = feat_k.reshape(batch_size, -1)

        K = len(self.queue)
        #         print('K',K)

        feat_k = feat_k.detach()

        # pos logit
        l_pos = F.cosine_similarity(feat_q, feat_k, dim=1)
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if K == 0 or not self.use_queue:
            l_neg = F.cosine_similarity(feat_q[:, None, :], feat_k[None, :, :], dim=-1)
        else:
            queue_tensor = torch.cat(self.queue, dim=0)
            print(queue_tensor.size())
            l_neg = F.cosine_similarity(feat_q[:, None, :], queue_tensor.reshape(-1, feat_q.size(1))[None, :, :],
                                        dim=-1)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  # batch_size * (K+1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        if self.use_queue:
            self.queue.append(feat_k.clone())
            #             for i in range(0,24):
            #                 if idx[i] not in self.idx_list and len(self.queue) <512:
            # #                     print(idx[i].item())
            # #                     print(self.idx_list)
            #                     self.idx_list.append(idx[i].item())
            #                     self.queue.append(feat_k[i].clone()[None,:])
            #                     print('LIST',len(self.idx_list))
            #                     print('1',feat_k[i][None,:].size())
            #                 elif idx[i] in self.idx_list:
            #                     print('duplicate')
            if K >= 512:
                #                 print('pop')
                self.queue.pop(0)
        #                 self.idx_list.pop(0)

        return loss

# class My_Loss(nn.Module):
#     def __init__(self, bdp_threshold, fdp_threshold,patch_num,temp=0.1, eps=1e-8):
#         super(My_Loss, self).__init__()
#         self.temp = temp
#         self.eps = eps
#         self.bdp_threshold = bdp_threshold
#         self.fdp_threshold = fdp_threshold
#         self.patch_num = patch_num
#         self.contrastive_loss = DRLoss(self.bdp_threshold,self.fdp_threshold,self.temp)
#         self.feature_bank = []
#         self.label_bank = []
#         self.FC_bank = []
#         self.count = 0
#         self.capacity = 10
#     def forward(self, uncertainty_map,reliable_map, entropy_map,thresh):
#
#         return loss_contr

# net = My_Loss(0.3,0.7,16)
# em_w = torch.randn(24,3,32,32)
# em_s = torch.randn(24,3,32,32)
# label = torch.randn(24,32,32)
# out = net(em_w,em_s,label)
# print(out)
# x = torch.randn(1,3,3)
# diagonal = torch.eye(3,dtype = bool)[None, :, :]
# x.masked_fill_(diagonal, -float('inf'))
# print(x)
class GaussianPosteriorPullLoss(nn.Module):
    def __init__(self, temperature=0.1, pull_strength=0.1):
        super().__init__()
        self.temp = temperature  # 控制概率锐化程度
        self.pull_strength = pull_strength  # 拉近强度系数

    def forward(self, a, proto_means, proto_vars):
        """
        参数:
            a: [N, dim]               - 输入特征
            proto_means: [K, dim]     - 类中心
            proto_vars: [K, dim]      - 各类各维度方差
        返回:
            loss: 标量
        """

        N, dim = a.shape
        K = proto_means.shape[0]

        # ===== 1. 计算对数概率密度 =====
        # 扩展维度用于广播计算 [N, K, dim]
        a_exp = a.unsqueeze(1).expand(-1, K, -1)
        means_exp = proto_means.unsqueeze(0).expand(N, -1, -1)
        vars_exp = proto_vars.unsqueeze(0).expand(N, -1, -1)
        safe_vars = proto_vars + 1e-6
        # 计算每个维度的对数概率 [N, K, dim]
        log_prob_per_dim = -0.5 * (
                torch.log(2 * torch.pi * safe_vars.unsqueeze(0)) +
                (a.unsqueeze(1) - proto_means.unsqueeze(0)) ** 2 / safe_vars.unsqueeze(0)
        )
        # 求和得到总对数概率 [N, K]
        log_prob = torch.sum(log_prob_per_dim, dim=2)

        # ===== 2. 计算后验概率 =====
        posterior = F.softmax(log_prob / self.temp, dim=1)  # [N, K]
        # ===== 3. 确定最可能类 =====
        max_prob, max_indices = torch.max(posterior, dim=1)  # [N]
        # ===== 4. 计算余弦相似度损失 =====
        # 归一化特征和类中心
        non_zero_mask = (a.norm(p=2, dim=1) > 1e-6)  # [N] Bool
        if non_zero_mask.sum() == 0:
            return torch.tensor(0.0, device=a.device, requires_grad=True)

        a_valid = a[non_zero_mask]
        a_norm = F.normalize(a_valid, p=2, dim=1)

        max_indices_valid = max_indices[non_zero_mask]
        max_prob_valid = max_prob[non_zero_mask]

        proto_norm = F.normalize(proto_means, p=2, dim=1)
        target_centers = proto_norm[max_indices_valid]

        cos_sim = torch.sum(a_norm * target_centers, dim=1)

        loss = (self.pull_strength * (1 - cos_sim) * max_prob_valid).mean()
        return loss

    def adjust_features(self, a, proto_means, proto_covs):
        """ 可选：直接返回调整后的特征 """
        with torch.no_grad():
            N = a.shape[0]
            log_prob_matrix = torch.zeros(N, proto_means.shape[0], device=a.device)
            for k in range(proto_means.shape[0]):
                diff = a - proto_means[k]
                inv_cov = torch.linalg.inv(proto_covs[k] + 1e-6 * torch.eye(a.shape[1], device=a.device))
                mahalanobis = torch.sum(diff @ inv_cov * diff, dim=1)
                log_det = torch.logdet(proto_covs[k] + 1e-6 * torch.eye(a.shape[1], device=a.device))
                log_prob_matrix[:, k] = -0.5 * (
                            a.shape[1] * torch.log(2 * torch.tensor(torch.pi)) + log_det + mahalanobis)
            posterior = F.softmax(log_prob_matrix / self.temp, dim=1)
            max_prob, max_indices = torch.max(posterior, dim=1)
            target_means = proto_means[max_indices]
            return a + 0.1 * (target_means - a)  # 移动步长0.1


class GaussianPosterior(nn.Module):
    def __init__(self, temperature=0.1, pull_strength=0.1):
        super().__init__()
        self.temp = temperature  # 控制概率锐化程度
        self.pull_strength = pull_strength  # 拉近强度系数

    def forward(self, a, proto_means, proto_vars, pseudo_label):
        """
        参数:
            a: [N, dim]               - 输入特征
            proto_means: [K, dim]     - 类中心
            proto_vars: [K, dim]      - 各类各维度方差
        返回:
            loss: 标量
        """

        N, dim = a.shape
        K = proto_means.shape[0]

        # ===== 1. 计算对数概率密度 =====
        # 扩展维度用于广播计算 [N, K, dim]
        a_exp = a.unsqueeze(1).expand(-1, K, -1)
        means_exp = proto_means.unsqueeze(0).expand(N, -1, -1)
        vars_exp = proto_vars.unsqueeze(0).expand(N, -1, -1)

        # 计算每个维度的对数概率 [N, K, dim]
        log_prob_per_dim = -0.5 * (
                torch.log(2 * torch.pi * proto_vars.unsqueeze(0)) +
                (a.unsqueeze(1) - proto_means.unsqueeze(0)) ** 2 / proto_vars.unsqueeze(0)
        )

        # 求和得到总对数概率 [N, K]
        log_prob = torch.sum(log_prob_per_dim, dim=2)

        # ===== 2. 计算后验概率 =====
        posterior = F.softmax(log_prob / self.temp, dim=1)  # [N, K]
        # ===== 3. 确定最可能类 =====
        max_prob, max_indices = torch.max(posterior, dim=1)  # [N]
        # ===== 4. 计算余弦相似度损失 =====
        # 归一化特征和类中心
        a_norm = F.normalize(a, p=2, dim=1)  # [N, dim]
        proto_norm = F.normalize(proto_means, p=2, dim=1)  # [K, dim]

        # 提取目标类中心 [N, dim]
        target_centers = proto_norm[max_indices]

        # 计算余弦相似度 [N]
        cos_sim = torch.sum(a_norm * target_centers, dim=1)

        # 加权损失
        loss = (self.pull_strength * (1 - cos_sim) * max_prob).mean()

        return loss

    def adjust_features(self, a, proto_means, proto_covs):
        """ 可选：直接返回调整后的特征 """
        with torch.no_grad():
            N = a.shape[0]
            log_prob_matrix = torch.zeros(N, proto_means.shape[0], device=a.device)
            for k in range(proto_means.shape[0]):
                diff = a - proto_means[k]
                inv_cov = torch.linalg.inv(proto_covs[k] + 1e-6 * torch.eye(a.shape[1], device=a.device))
                mahalanobis = torch.sum(diff @ inv_cov * diff, dim=1)
                log_det = torch.logdet(proto_covs[k] + 1e-6 * torch.eye(a.shape[1], device=a.device))
                log_prob_matrix[:, k] = -0.5 * (
                            a.shape[1] * torch.log(2 * torch.tensor(torch.pi)) + log_det + mahalanobis)
            posterior = F.softmax(log_prob_matrix / self.temp, dim=1)
            max_prob, max_indices = torch.max(posterior, dim=1)
            target_means = proto_means[max_indices]
            return a + 0.1 * (target_means - a)  # 移动步长0.1


class MarginWeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MarginWeightedCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, margin=None, softmax=True):

        if softmax:
            inputs = F.log_softmax(inputs, dim=1)  # 使用 log_softmax 搭配 nll_loss
            loss_map = F.nll_loss(inputs, targets, reduction='none')  # [B, H, W]
        else:
            loss_map = F.cross_entropy(inputs, targets, reduction='none')  # [B, H, W]

        if margin is not None:
            margin = margin.float()
            loss = (margin * loss_map).mean()
        else:
            loss = loss_map.mean()

        return loss