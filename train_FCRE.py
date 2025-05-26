import argparse
import logging
import os
import random
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils_.losses import ConLoss,PL_loss, DR_loss
from config import get_config
import math
from dataset import (
    BaseDataSets,
    CTATransform,
    RandomGenerator,
    TwoStreamBatchSampler,
    WeakStrongAugment,
    RandomGenerator_w,
)
from networks.net_factory import net_factory
from utils_ import losses, metrics, ramps, util,feature_memory
from val_2D import test_single_volume
from torch.distributions import Categorical
import augmentations

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC_new/JointCPS', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_F', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')

parser.add_argument('--kernel_size', type=int, default=8,
                    help='size of patches')
parser.add_argument("--load", default=True, action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.95,
    help="confidence threshold for using pseudo-labels",
)
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per epoch')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency1', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency2', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

parser.add_argument(
    '--cfg', type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "130": 1132, "126": 1058, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

# model1 = ViT_seg(config, img_size=args.patch_size,
#                  num_classes=args.num_classes).cuda()
# model1.load_from(config)

def get_current_consistency_weight(consistency, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def find_uncertain_positions(features, proto_means, proto_vars, temp=0.9):
    """
    找出最大后验概率与次大后验概率差值小于阈值的位置

    参数:
        features: [5,16,224,224] 的张量，表示特征图
        proto_means: [K,16] 的张量，表示K个类别的中心
        proto_vars: [K,16] 的张量，表示K个类别的方差
        temp: 温度参数，控制概率锐化程度

    返回:
        uncertain_mask: [5,224,224] 的布尔张量，True表示差值小于阈值的位置
    """
    B, C, H, W = features.shape  # B=5, C=16, H=224, W=224
    K = proto_means.shape[0]  # 类别数

    # 重塑特征以便于计算
    features_reshaped = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
    features_reshaped = F.normalize(features_reshaped, p=2, dim=1)

    # 计算对数概率密度
    log_probs = torch.zeros(B * H * W, K, device=features.device)

    for k in range(K):
        # 计算每个维度的对数概率 [B*H*W, C]
        log_prob_per_dim = -0.5 * (
                torch.log(2 * torch.pi * proto_vars[k]) +
                (features_reshaped - proto_means[k]) ** 2 / proto_vars[k]
        )

        # 求和得到总对数概率 [B*H*W]
        log_probs[:, k] = torch.sum(log_prob_per_dim, dim=1)

    # 计算后验概率
    posterior = F.softmax(log_probs / temp, dim=1)  # [B*H*W, K]

    # 获取每个位置的最大和次大概率值
    top2_probs, top2_classes = torch.topk(posterior, k=2, dim=1)  # [B*H*W, 2]

    # 计算最大与次大概率的差
    prob_diff = top2_probs[:, 0] - top2_probs[:, 1]  # [B*H*W]
    margin_confidence = prob_diff.reshape(B,H,W)

    return posterior,margin_confidence

def asymmetric_kl_consistency_loss(pa, pb, ma, mb, epsilon=1e-6):
    """
    pa, pb: [B, C, D, H, W] - softmax 概率输出
    ma, mb: [B, 1, D, H, W] - 置信权重（A和B方向）
    返回: 散度加权的一致性损失（标量）
    """
    ma = ma.unsqueeze(1)  # [B, 1, D, H, W] -> broadcast to C
    mb = mb.unsqueeze(1)

    # 稳定性处理
    pa = pa.clamp(min=epsilon)
    pb = pb.clamp(min=epsilon)

    # 平均分布 M
    m = 0.5 * (pa + pb)

    # KL(P || M) 和 KL(Q || M)
    kl_pa_m = pa * (torch.log(pa) - torch.log(m))  # [B, C, D, H, W]
    kl_pb_m = pb * (torch.log(pb) - torch.log(m))

    # JS 散度是两者平均
    js_div = 0.5 * kl_pa_m + 0.5 * kl_pb_m  # [B, C, D, H, W]

    # 加权求平均
    weight = (ma + mb) / 2  # 对 A 和 B 的 confidence 平均处理
    loss = (js_div * weight).sum() / (weight.sum() + epsilon)

    return loss
def get_dynamic_loss_weight(epoch, total_epochs, growth="sigmoid"):
    """
    根据当前 epoch 动态调整损失权重。

    参数:
    - epoch: 当前训练的 epoch
    - total_epochs: 训练的总 epoch 数
    - growth: 增长方式，"linear" 为线性，"sigmoid" 为 sigmoid 增长

    返回:
    - weight: 当前 epoch 的损失权重，范围为 [0, 1]
    """
    if growth == "linear":
        # 线性增长：权重从 0 增加到 1
        weight = min(1.0, epoch / total_epochs)
    elif growth == "sigmoid":
        # 使用 sigmoid 函数平滑增长，增长速度较慢
        k = 10  # 控制增长速度的参数，可以调整
        weight = 1 / (1 + math.exp(-k * (epoch / total_epochs - 0.5)))
    else:
        raise ValueError("Unsupported growth type. Choose 'linear' or 'sigmoid'.")

    return weight

def unreliable_map(posterior_probs, softmax_probs):
    """
    margin_confidence: [B, C, H, W]
    posterior_probs:   [B, C, H, W]
    softmax_probs:     [B, C, H, W]
    features:          [B, C_feat, H, W]
    class_centers:     [num_classes, C_feat]
    """
    B, C,H, W = softmax_probs.shape

    # 1. 获取 softmax 和 posterior 的 argmax 类别
    softmax_label = torch.argmax(softmax_probs, dim=1)   # [B,H,W]
    posterior_label = torch.argmax(posterior_probs,dim=1).reshape(B,H,W)  # [B,H,W]
    # print(softmax_label.shape,posterior_label.shape)
    # 2. 构造不可靠区域掩码
    unreliable_mask = (softmax_label != posterior_label).float()  # [B,H, W]
    return unreliable_mask
def margin_based_uncertainty_loss(feature, post, margin_confidence, mask, class_centers):
    """
    feature: (B, dim, D, H, W)
    post: (B, C, D, H, W)
    margin_confidence: (B, D, H, W)
    mask: (B, D, H, W)
    class_centers: (C, dim)
    """
    B, dim,H, W = feature.shape
    C = class_centers.shape[0]

    # 1. 获取后验概率最大类索引（B, D, H, W）
    top_class = torch.argmax(post, dim=1).reshape(B,H,W)  # shape: (B, D, H, W)

    # 2. 转置 feature → (B, D, H, W, dim)
    feature = feature.permute(0, 2, 3, 1).contiguous()

    # 3. 展平
    feature_flat = feature.view(-1, dim)              # (B*D*H*W, dim)
    top_class_flat = top_class.view(-1)               # (B*D*H*W,)
    mask_flat = mask.view(-1).bool()                  # (B*D*H*W,)
    margin_flat = margin_confidence.view(-1)          # (B*D*H*W,)

    # 4. 仅保留不可靠区域
    feature_sel = feature_flat[mask_flat]             # (N, dim)
    top_class_sel = top_class_flat[mask_flat]         # (N,)
    margin_sel = margin_flat[mask_flat]               # (N,)

    if feature_sel.numel() == 0:
        return torch.tensor(0.0, device=feature.device, requires_grad=True)

    # 5. 获取对应的 class_center → (N, dim)
    centers_sel = class_centers[top_class_sel]        # (N, dim)

    # 6. 计算欧氏距离 or cosine（这里用 L2）
    dist = F.mse_loss(feature_sel, centers_sel, reduction='none')  # (N, dim)
    dist = dist.mean(dim=1)                                        # (N,)

    # 7. 权重：1 - margin（置信度小 → 权重大）
    weights = 1.0 - margin_sel                                     # (N,)
    weighted_loss = (dist * weights).mean()

    return weighted_loss



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def compute_class_priors(pseudo_labels, num_classes):
    """
    计算各类别先验概率：(伪标签中属于该类的像素数) / 总像素数

    参数:
        pseudo_labels: [N, H, W] 伪标签矩阵（值为0~K-1的整数）
        num_classes: 类别数K
    返回:
        class_priors: [K] 各类别先验概率
    """
    # 统计各类像素数
    class_counts = torch.bincount(pseudo_labels, minlength=num_classes)

    # 计算概率（避免除零）
    total_pixels = pseudo_labels.numel()
    priors = class_counts.float() / (total_pixels + 1e-6)

    # 归一化确保总和为1
    return priors / (priors.sum() + 1e-6)


def train(args, snapshot_path):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    intra_loss = losses.GaussianPosteriorPullLoss()
    #     loss_type = 'MT_loss'
    def power_law_sharpening(tensor, gamma=0.99):
        """
        实现公式: Y'_j = (Y_j^gamma) / (Y_j^gamma + (1-Y_j)^gamma)
        适用于torch.Tensor格式，维度为(B,C,H,W)

        参数:
            tensor: 输入张量，维度为(B,C,H,W)，值域应在[0,1]之间
            gamma: 幂指数，默认为0.5

        返回:
            锐化后的张量，同样维度为(B,C,H,W)
        """
        # 确保输入值在[0,1]之间
        tensor = torch.clamp(tensor, 0, 1)

        # 计算分子
        numerator = torch.pow(tensor, gamma)

        # 计算分母
        denominator = torch.pow(tensor, gamma) + torch.pow(1 - tensor, gamma)

        # 应用变换
        sharpened = numerator / denominator

        return sharpened


    def make_prototype_center_all_cross_attention(features, mask, num_class, bank, other_bank, topk=5000):
        b, c, h, w = features.size()
        features = features.view(b, c, -1)  # [B, C, HW]
        mask = mask.view(b, -1)  # [B, HW]

        class_means = torch.zeros([num_class, c]).to(features.device)
        class_counts = torch.zeros([num_class]).to(features.device)
        class_vars = torch.zeros([num_class, c]).to(features.device)

        for cls in range(num_class):
            # step 1: 获取当前 batch 中该类特征
            mask_cls = (mask == cls)  # [B, HW]
            selected_features = features.permute(0, 2, 1)[mask_cls]  # [N, C]

            # if selected_features.size(0) == 0:
            #     selected_features = bank.get_all(cls).squeeze(0).to(features.device)
            #     continue

            # step 2: 当前特征入自己的 bank
            bank.push(selected_features, cls)
            selected_features = bank.get_all(cls).squeeze(0).to(features.device)
            if selected_features.size(0) == 0:
                continue
            # # step 3: 获取另一个子网中的 memory bank 特征
            # memory_B = other_bank.get_all(cls).squeeze(0).to(features.device)  # [M, C]
            # # print(memory_B.size(0))
            # # print(selected_features.shape, memory_B.shape)
            # if memory_B.size(0) < topk:
            #     # case 1: 没有 enough 特征或为空，直接用自己的特征计算
            #     top_features = selected_features
            # else:
            #     # case 2: memory bank 中该类已有数据，使用交叉注意力选 topk 对B来说A中最重要的K个特征
            #     query = memory_B  # [M, C]
            #     key = selected_features  # [N, C]
            #     value = selected_features  # [N, C]
            #
            #     attn_scores = torch.matmul(query, key.T) / (c ** 0.5 * 0.07)  # [M, N]
            #     attn_weights = F.softmax(attn_scores, dim=1)  # [M, N]
            #
            #     mean_attn = attn_weights.mean(dim=0)  # [N]
            #     topk_idx = torch.topk(mean_attn, topk).indices  # [topk]
            #     top_features = value[topk_idx]  # [topk, C]

            # step 4: 计算均值和方差
            cur_mean = torch.mean(selected_features, dim=0)
            cur_count = selected_features.size(0)
            diff = selected_features - cur_mean.unsqueeze(0)
            variance = torch.mean(diff ** 2, dim=0)

            # step 5: 存储

            class_means[cls] = cur_mean
            class_counts[cls] = cur_count
            class_vars[cls] = variance

        return class_means, class_counts, class_vars
    def create_model(net_type,in_ch,ema=False):
        # Network definition
        model = net_factory(net_type=net_type, in_chns=in_ch,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model(args.model,in_ch=1)
    model2 = create_model(args.model,in_ch=1)
    memory_bank_label1 = feature_memory.MemoryBank(max_size=20000,num_classes=4)
    memory_bank_label2 = feature_memory.MemoryBank(max_size=20000,num_classes=4)

    pl_loss = PL_loss()
    dr_loss = DR_loss()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def get_comp_loss(weak, strong):
        """get complementary loss and adaptive sample weight.
        Compares least likely prediction (from strong augment) with argmin of weak augment.
        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch
        Returns:
            comp_loss, as_weight
        """
        il_output = torch.reshape(
            strong,
            (
                args.batch_size,
                args.num_classes,
                args.patch_size[0] * args.patch_size[1],
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.patch_size[0] * args.patch_size[1]))
        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(
            torch.add(torch.negative(strong), 1),
            comp_labels,
        )
        return comp_loss, as_weight

    def margin_based_consistency_loss(f1, f2, prob, lambda_reg=0.1):
        """
        f1, f2: [N, D, H, W] — 两个模型输出的特征
        prob: [N, C, H, W] — 预测类别的 softmax 概率
        """
        # 1. 计算 margin: max - 2nd max
        margin = prob
        # 2. 计算特征差距
        diff = F.mse_loss(f1, f2, reduction='none')  # [N, D, H, W]
        diff = diff.mean(dim=1)  # → [N, H, W]

        # 3. 计算主损失（不确定性加权一致性）
        main_loss = (margin * diff).mean()

        # 4. margin 正则项（鼓励 margin 整体变大）
        reg_loss = (1 - margin).mean()

        # 5. 总损失
        total_loss = main_loss + lambda_reg * reg_loss
        return total_loss

    def make_entropy(tensor):
        b,c,h,w = tensor.size(0),tensor.size(1),tensor.size(2) ,tensor.size(3)
        # prob_tensor = F.softmax(tensor,dim=1)
        prob_tensor = tensor
        entropy = -torch.sum(prob_tensor * torch.log2(prob_tensor + 1e-10),dim=1)

        return entropy

    def get_feature(features, mask):
        b, c, h, w = features.shape
        selected_features = features.permute(0, 2,3,1)[mask]  # N,dim
        return selected_features

    def compute_maximization_loss(mu):
        """
        计算最大化损失 L_max。
        参数:
            mu (Tensor): 形状为 (K, D) 的张量，K为类别数，D为均值向量的维度。
        返回:
            Tensor: 标量损失值。
        """
        K = mu.size(0)
        if K < 2:
            return torch.tensor(0.0, device=mu.device)

        # 计算所有成对差值 (K, K, D)
        diff = mu.unsqueeze(1) - mu.unsqueeze(0)
        # 计算平方距离 (K, K)
        sq_dist = torch.sum(diff ** 2, dim=2)
        # 应用指数函数
        exp_terms = torch.exp(-sq_dist)

        # 排除对角线元素（k ≠ σ）
        mask = ~torch.eye(K, dtype=torch.bool, device=mu.device)
        valid_terms = exp_terms[mask]

        # 计算损失值
        loss = (2.0 / (K * (K - 1))) * valid_terms.sum()
        return loss

    def refresh_policies(db_train, cta, random_depth_weak, random_depth_strong):
        db_train.ops_weak = cta.policy(probe=False, weak=True)
        db_train.ops_strong = cta.policy(probe=False, weak=False)
        cta.random_depth_weak = random_depth_weak
        cta.random_depth_strong = random_depth_strong
        if max(Counter([a.f for a in db_train.ops_weak]).values()) >= 3 or max(
                Counter([a.f for a in db_train.ops_strong]).values()) >= 3:
            print('too deep with one transform, refresh again')
            refresh_policies(db_train, cta, random_depth_weak, random_depth_strong)
        logging.info(f"CTA depth weak: {cta.random_depth_weak}")
        logging.info(f"CTA depth strong: {cta.random_depth_strong}")
        logging.info(f"\nWeak Policy: {db_train.ops_weak}")
        #         logging.info(f"\nWeak Policy: {max(Counter([a.f for a in db_train.ops_weak]).values())}")
        logging.info(f"Strong Policy: {db_train.ops_strong}")


    cta = augmentations.ctaugment.CTAugment()
    transform = CTATransform(args.patch_size, cta)

    # sample initial weak and strong augmentation policies (CTAugment)
    ops_weak = cta.policy(probe=False, weak=True)
    ops_strong = cta.policy(probe=False, weak=False)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transform,
        ops_weak=ops_weak,
        ops_strong=ops_strong,
    )

    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
    )

    model1.train()
    model2.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    # optimizer1 = optim.Adam(model1.parameters(), lr=base_lr,
    #                        weight_decay=0.0001)
    # optimizer2 = optim.Adam(model2.parameters(), lr=base_lr,
    #                        weight_decay=0.0001)

    iter_num = 0
    start_epoch = 0

    ce_loss = CrossEntropyLoss()
    ACE_loss = losses.MarginWeightedCrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    pixel_wise_contrastive_loss_criter = ConLoss()
    # contrastive_loss_sup_criter = contrastive_loss_sup()
    contrastive_loss_sup_criter = ConLoss()
    # contrastive_loss_sup_criter = My_Loss(0.3,0.7,64)
    # infoNCE_loss = InfoNCE()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    lr_ = base_lr
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    a0 = 0.2
    r_loss_seg_dice_mean = []
    v_loss_seg_dice_mean = []
    epoch_number = []
    for epoch_num in iterator:
        r_loss_seg_dice_total = 0.0
        v_loss_seg_dice_total = 0.0
        a1 = 0.01
        epoch_errors = []
        if iter_num <= 10000:
            random_depth_weak = np.random.randint(3, high=5)
            random_depth_strong = np.random.randint(2, high=5)
        elif iter_num >= 20000:
            random_depth_weak = 2
            random_depth_strong = 2
        else:
            random_depth_weak = np.random.randint(2, high=5)
            random_depth_strong = np.random.randint(2, high=5)
        refresh_policies(db_train, cta, random_depth_weak, random_depth_strong)

        running_loss = 0.0
        running_sup_loss = 0.0
        running_unsup_loss = 0.0
        running_comple_loss = 0.0
        running_con_l_l = 0
        running_con_l_u = 0
        running_con_loss = 0
        for i_batch, sampled_batch in enumerate(zip(trainloader)):
            raw_batch, weak_batch, strong_batch, label_batch_aug, label_batch = (
                sampled_batch[0]["image"],
                sampled_batch[0]["image_weak"],
                sampled_batch[0]["image_strong"],
                sampled_batch[0]["label_aug"],
                sampled_batch[0]["label"],
            )

            label_batch_aug[label_batch_aug >= 4] = 0
            label_batch_aug[label_batch_aug < 0] = 0
            weak_batch, strong_batch, label_batch_aug = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch_aug.cuda(),
            )

            # handle unfavorable cropping
            non_zero_ratio = torch.count_nonzero(label_batch) / (24 * 224 * 224)
            # non_zero_ratio = 0.001
            non_zero_ratio_aug = torch.count_nonzero(label_batch_aug) / (24 * 224 * 224)
            #             print(label_batch.unique(return_counts=True))
            #             print(non_zero_ratio)
            if non_zero_ratio > 0 and non_zero_ratio_aug < 0.005:  # try 0.01
                logging.info("Refreshing policy...")
                refresh_policies(db_train, cta, random_depth_weak, random_depth_strong)
            #################################################################################################################################
            # outputs for model
            outputs_weak1_logits,features1 = model1(weak_batch)
            outputs_weak_soft1 = torch.softmax(outputs_weak1_logits, dim=1)
            outputs_strong1,feature1_strong = model1(strong_batch)
            outputs_strong_soft1 = torch.softmax(outputs_strong1, dim=1)

            outputs_weak2_logits,features2 = model2(weak_batch)
            outputs_weak_soft2 = torch.softmax(outputs_weak2_logits, dim=1)
            outputs_strong2,feature2_strong = model2(strong_batch)
            outputs_strong_soft2 = torch.softmax(outputs_strong2, dim=1)
            #################################################################################################################################
            # supervised loss
            sup_loss1 = dice_loss(
                outputs_weak_soft1[: args.labeled_bs],
                label_batch_aug[: args.labeled_bs].unsqueeze(1),
            )

            sup_loss2 = dice_loss(
                outputs_weak_soft2[: args.labeled_bs],
                label_batch_aug[: args.labeled_bs].unsqueeze(1),
            )

            supervised_loss = sup_loss1 + sup_loss2
            pseudo_label1 = torch.argmax(outputs_weak_soft1[args.labeled_bs:].detach(),dim=1,keepdim=False)
            pseudo_label2 = torch.argmax(outputs_weak_soft2[args.labeled_bs:].detach(),dim=1,keepdim=False)
            # print(pseudo_label1.shape,outputs_weak_soft2[args.labeled_bs:].shape)


            center1, counts1, var1 = make_prototype_center_all_cross_attention(features1[:args.labeled_bs],
                                                               label_batch_aug[:args.labeled_bs], num_classes,
                                                               memory_bank_label1,memory_bank_label2)


            center2, counts2, var2 = make_prototype_center_all_cross_attention(features2[:args.labeled_bs],
                                                               label_batch_aug[:args.labeled_bs], num_classes,
                                                               memory_bank_label2,memory_bank_label1)



            var_los1 = compute_maximization_loss(center1)
            var_los2 = compute_maximization_loss(center2)
            var_loss = var_los1 + var_los2

            post1,prob1 = find_uncertain_positions(features1[args.labeled_bs:],center2,var2)  #5,224,224
            post2,prob2 = find_uncertain_positions(features2[args.labeled_bs:],center1,var1)  #5,224,224
            margin_loss = asymmetric_kl_consistency_loss(outputs_weak_soft1[args.labeled_bs:],outputs_weak_soft2[args.labeled_bs:],prob1,prob2)
            loss_consistency = margin_loss

            unsupervised_loss1 = dice_loss(outputs_weak_soft2[args.labeled_bs:],pseudo_label1.unsqueeze(1),margin=prob1) + ACE_loss(outputs_weak2_logits[args.labeled_bs:],pseudo_label1,margin=prob1)
            unsupervised_loss2 = dice_loss(outputs_weak_soft1[args.labeled_bs:],pseudo_label2.unsqueeze(1),margin=prob2) + ACE_loss(outputs_weak1_logits[args.labeled_bs:],pseudo_label2,margin=prob2)
            loss_mwcps = unsupervised_loss1 + unsupervised_loss2

            unreliable_region1 = unreliable_map(post1,outputs_weak_soft1[args.labeled_bs:])
            unreliable_region2 = unreliable_map(post2,outputs_weak_soft2[args.labeled_bs:])
            intra_loss1 = margin_based_uncertainty_loss(features1[args.labeled_bs:],post1,prob1,unreliable_region1,center1)
            intra_loss2 = margin_based_uncertainty_loss(features2[args.labeled_bs:],post2,prob2,unreliable_region2,center2)
            intra_loss_total = intra_loss1 + intra_loss2
            print(intra_loss_total.item())


            if iter_num < 0:
                consistency_weight2 = 0
                consistency_weight1 = 0
            else:
                consistency_weight2 = get_current_consistency_weight(args.consistency2,
                                                                     iter_num // 150)
                consistency_weight1 = get_current_consistency_weight(args.consistency1,
                                                                     iter_num // 150)

            # both
            #             loss = sup_loss + consistency_weight1 * (Loss_contrast_l + unsup_loss + consistency_weight2 *  Loss_contrast_u)
            loss = supervised_loss + var_loss * consistency_weight2 * 0.05 + loss_mwcps * consistency_weight1 + intra_loss_total * consistency_weight2 + 0.05
            # loss = supervised_loss + var_loss * consistency_weight2 + consistency_weight2 * loss_consistency + unsupervised_loss * consistency_weight2
            # print(ambiguity1_features.shape,ambiguity2_features.shape)
            # print(intra_loss1.item(),intra_loss2.item())

            #             loss = 0.5 * (sup_loss + consistency_weight2 * unsup_loss + consistency_weight2 * contrastive_loss)
            # make_dot(loss,params=dict(model1.named_parameters())).render('model1',format="png")

            # running_loss += loss
            # running_sup_loss += sup_loss
            # running_unsup_loss += unsup_loss
            # #             running_comple_loss += comple_loss
            # running_con_loss += contrastive_loss
            # running_con_l_l += Loss_contrast_l
            # running_con_l_u += Loss_contrast_u

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()


            # track batch-level error, used to update augmentation policy
            epoch_errors.append(0.5 * loss.item())

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            writer.add_scalar("lr", lr_, iter_num)
            # writer.add_scalar("consistency_weight/consistency_weight1", consistency_weight1, iter_num)
            writer.add_scalar("consistency_weight/consistency_weight2", consistency_weight2, iter_num)
            writer.add_scalar("loss/consistency_loss", loss_consistency.item(), iter_num)
            # writer.add_scalar("loss/contrastive_loss", loss_DR, iter_num)
            logging.info("iteration %d : model loss : %f" % (iter_num, loss.item()))

            if iter_num % 50 == 0:
                idx = args.labeled_bs
                # show raw image
                # image = raw_batch[idx, 0:1, :, :]
                # writer.add_image("train/RawImage", image, iter_num)
                # show weakly augmented image
                image = weak_batch[idx, 0:1, :, :]
                writer.add_image("train/WeakImage", image, iter_num)
                # show strongly augmented image
                image_strong = strong_batch[idx, 0:1, :, :]
                writer.add_image("train/StrongImage", image_strong, iter_num)
                # show model prediction (strong augment)
                outputs_strong1 = torch.argmax(outputs_strong_soft1, dim=1, keepdim=True)
                writer.add_image("train/model_Prediction1", outputs_strong1[idx, ...] * 50, iter_num)
                outputs_strong2 = torch.argmax(outputs_strong_soft2, dim=1, keepdim=True)
                writer.add_image("train/model_Prediction2", outputs_strong2[idx, ...] * 50, iter_num)
                # show ground truth label
                labs = label_batch_aug[idx, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)
                # show generated pseudo label
                # pseudo_labs1 = pseudo_outputs1[idx, ...].unsqueeze(0) * 50
                # writer.add_image("train/PseudoLabel1", pseudo_labs1, iter_num)
                # pseudo_labs2 = pseudo_outputs2[idx, ...].unsqueeze(0) * 50
                # writer.add_image("train/PseudoLabel2", pseudo_labs2, iter_num)
                # pseudo_labs = pseudo_outputs[idx, ...].unsqueeze(0) * 50
                # writer.add_image("train/PseudoLabel", pseudo_labs, iter_num)

            if iter_num > 0 and iter_num % 1000 == 0:
                model1.eval()
                metric_list = 0.0
                for _, sampled_batch in (enumerate
                    (valloader)):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('eval/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('eval/model1_val_mean_hd95',
                                  mean_hd951, iter_num)
                print('dice1:', performance1)
                print('mean_hd1:', mean_hd951)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    if performance1 > 0:
                        save_mode_path = os.path.join(snapshot_path,
                                                      'model1_iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance1, 4)))
                        save_best = os.path.join(snapshot_path,
                                                 '{}_best_model1.pth'.format(args.model))
                        # save_proj = os.path.join(snapshot_path,'{}_best_projection5.pth'.format(args.model))
                        memory_bank_label1.save(os.path.join(snapshot_path,
                                                             'bank1_iter_{}_dice_{}.pth'.format(
                                                                 iter_num, round(best_performance1, 4))))
                        util.save_checkpoint(epoch_num, model1, optimizer1, loss, save_mode_path)
                        util.save_checkpoint(epoch_num, model1, optimizer1, loss, save_best)
                        # util.save_checkpoint(epoch_num, projector_5, optimizer1,loss,path=save_proj)
                        # util.save_checkpoint(epoch_num, model1, optimizer1, projector_1, projector_3, cta,
                        #                      best_performance1, save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('eval/model2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('eval/model2_val_mean_hd95',
                                  mean_hd952, iter_num)
                print('dice2:', performance2)
                print('mean_hd2:', mean_hd952)
                if performance2 > best_performance2:
                    best_performance2 = performance2
                    if performance2 > 0:
                        save_mode_path = os.path.join(snapshot_path,
                                                      'model2_iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance2, 4)))
                        save_best = os.path.join(snapshot_path,
                                                 '{}_best_model2.pth'.format(args.model))
                        # save_proj = os.path.join(snapshot_path, '{}_best_projection5.pth'.format(args.model))
                        memory_bank_label2.save(os.path.join(snapshot_path,
                                                      'bank2_iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance2, 4))))
                        util.save_checkpoint(epoch_num, model2, optimizer2, loss, save_mode_path)
                        util.save_checkpoint(epoch_num, model2, optimizer2, loss, save_best)
                        # util.save_checkpoint(epoch_num, projector_5, optimizer1, loss, path=save_proj)
                        # util.save_checkpoint(epoch_num, model2, optimizer2, projector_2, projector_4, cta,
                        #                      best_performance2, save_mode_path)
                        # util.save_checkpoint(epoch_num, model2, optimizer2, projector_2, projector_4, cta,
                        #                      best_performance2, save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()
                logging.info(
                    'current best dice coef model 1 {}, model 2 {}'.format(best_performance1, best_performance2))

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')

                #                 util.save_checkpoint(epoch_num, model1, optimizer1, loss, save_mode_path)
                # util.save_checkpoint(epoch_num, model1, optimizer1, projector_1, projector_3, cta, best_performance1,
                #                      save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')

                #                 util.save_checkpoint(epoch_num, model2, optimizer2, loss, save_mode_path)
                # util.save_checkpoint(epoch_num, model2, optimizer2, projector_2, projector_4, cta, best_performance2,
                #                      save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        r_loss_seg_dice_mean.append(r_loss_seg_dice_total / len(trainloader))
        v_loss_seg_dice_mean.append(v_loss_seg_dice_total / len(trainloader))
        # epoch_number.append(epoch_num)
        # if epoch_num == 50:
        #     plt.plot(epoch_number, r_loss_seg_dice_mean, label='net1_loss')
        #     plt.plot(epoch_number, v_loss_seg_dice_mean,label='net2_loss')
        #     plt.xlabel("epoch")
        #     plt.ylabel("los")
        #     plt.legend()
        #     plt.savefig("with_competition.png", format="png", dpi=300)
        if iter_num >= max_iterations:
            iterator.close()
            break

        epoch_loss = running_loss / len(trainloader)
        epoch_sup_loss = running_sup_loss / len(trainloader)
        epoch_unsup_loss = running_unsup_loss / len(trainloader)
        #         epoch_comple_loss = running_comple_loss / len(trainloader)
        epoch_con_loss = running_con_loss / len(trainloader)
        epoch_con_loss_u = running_con_l_u / len(trainloader)
        epoch_con_loss_l = running_con_l_l / len(trainloader)

        logging.info('{} Epoch [{:03d}/{:03d}]'.
                     format(datetime.now(), epoch_num, max_epoch))
        logging.info('Train loss: {}'.format(epoch_loss))
        writer.add_scalar('Train/Loss', epoch_loss, epoch_num)

        logging.info('Train sup loss: {}'.format(epoch_sup_loss))
        writer.add_scalar('Train/sup_loss', epoch_sup_loss, epoch_num)

        logging.info('Train unsup loss: {}'.format(epoch_unsup_loss))
        writer.add_scalar('Train/unsup_loss', epoch_unsup_loss, epoch_num)

        #         logging.info('Train comple loss: {}'.format(epoch_comple_loss))
        #         writer.add_scalar('Train/comple_loss', epoch_comple_loss, epoch_num)

        logging.info('Train contrastive loss: {}'.format(epoch_con_loss))
        writer.add_scalar('Train/contrastive_loss', epoch_con_loss, epoch_num)

        # logging.info('Train weighted contrastive loss: {}'.format(
        #     consistency_weight1 * Loss_contrast_l + consistency_weight2 * Loss_contrast_u))
        # writer.add_scalar('Train/weighted_contrastive_loss',
        #                   consistency_weight1 * Loss_contrast_l + consistency_weight2 * Loss_contrast_u, epoch_num)

        logging.info('Train contrastive loss l: {}'.format(epoch_con_loss_l))
        writer.add_scalar('Train/contrastive_loss_l', epoch_con_loss_l, epoch_num)

        logging.info('Train contrastive loss u: {}'.format(epoch_con_loss_u))
        writer.add_scalar('Train/contrastive_loss_u', epoch_con_loss_u, epoch_num)

        # update policy parameter bins for sampling
        mean_epoch_error = np.mean(epoch_errors)
        cta.update_rates(db_train.ops_weak, 1.0 - 0.5 * mean_epoch_error)
        cta.update_rates(db_train.ops_strong, 1.0 - 0.5 * mean_epoch_error)

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    # if not args.deterministic:
    #     cudnn.benchmark = True
    #     cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    snapshot_path = "model/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
    # if not os.path.exists(snapshot_path):
    #     os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    #
    # logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))

    train(args, snapshot_path)
    # model1 = ViT_seg(config, img_size=args.patch_size,
    #                  num_classes=args.num_classes).cuda()