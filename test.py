import argparse
import os
import shutil
import cv2
import h5py
from utils_.feature_memory import MemoryBank
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import torch.nn.functional as F
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage.transform import resize
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='just_test', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_F', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
bank1 = MemoryBank(max_size=20000, num_classes=4)
bank1.load('E:\workspace\Semi-Supervised\JointCPS_7_labeled/unet_F/bank1_iter_4000_dice_0.7398.pth')
bank2 = MemoryBank(max_size=20000, num_classes=4)
bank2.load('E:\workspace\Semi-Supervised\JointCPS_7_labeled/unet_F/bank1_iter_27000_dice_0.8901.pth')


def confident_incorrect_ratio(pseudo_label, true_label, entropy_map, threshold=0.75):
    """
    返回：伪标签中，被认为可靠但预测错误的像素占**总像素数**的比例。
    """
    # 预测错误的位置
    incorrect_mask = (pseudo_label != true_label)  # bool (B, H, W)

    # 低不确定性区域（可靠区域）
    # reliable_mask = (entropy_map < threshold)  # bool (B, H, W)
    reliable_mask = (entropy_map)  # bool (B, H, W)

    # 可靠但错误的像素
    confident_incorrect = incorrect_mask & reliable_mask  # bool

    # 像素统计
    total_pixels = incorrect_mask.sum()
    confident_wrong = confident_incorrect.sum().item()

    return confident_wrong / total_pixels
def enable_dropout(model):
    """强制打开 dropout"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.train()


def mc_dropout_uncertainty_entropy(model, input_tensor, T=8, return_mean=False):
    """
    Monte Carlo Dropout 不确定性（基于熵）

    参数：
        model: 含 Dropout 的模型
        input_tensor: (B, C, H, W)
        T: MC前向次数
        return_mean: 是否返回平均预测结果
    返回：
        - 若 return_mean=False: 只返回熵不确定性 (B, H, W)
        - 若 return_mean=True: 返回 (平均预测, 熵不确定性)
    """
    model.eval()
    enable_dropout(model)  # 强制启用 Dropout 层

    preds = []

    with torch.no_grad():
        for _ in range(T):
            output, _ = model(input_tensor)  # 假设模型返回 (logits, features)
            prob = F.softmax(output, dim=1)  # (B, C, H, W)
            preds.append(prob)

    preds = torch.stack(preds, dim=0)  # (T, B, C, H, W)
    mean_pred = preds.mean(dim=0)  # (B, C, H, W)

    # 熵作为不确定性
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-6), dim=1)  # (B, H, W)

    if return_mean:
        return mean_pred, entropy
    else:
        return entropy
def save_uncertainty_overlay(image, uncertainty, save_path, alpha=0.6, cmap='JET'):
    """
    image: Tensor (H, W), grayscale image
    uncertainty: Tensor (H, W)
    save_path: str, path to save the overlay image (e.g., 'result/overlay.png')
    alpha: float, heatmap transparency
    cmap: str, OpenCV colormap (e.g., 'JET', 'HOT')
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转 numpy
    image = image
    uncertainty = uncertainty

    # 归一化 image 到 [0, 255]
    if image.max() <= 1:
        image = image * 255
    image = image.astype(np.uint8)

    # 转为3通道灰度图
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 归一化不确定性为 heatmap
    unc = np.log1p(uncertainty)  # log平滑（可防止高值主导）
    unc = (unc - unc.min()) / (unc.max() - unc.min() + 1e-6)
    unc_map = (unc * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(unc_map, getattr(cv2, f'COLORMAP_{cmap.upper()}'))

    # 叠加热图
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)

    # 保存为图像文件
    cv2.imwrite(save_path, overlay)

def make_prototype_center(features,mask,num_class):
    b, c, h, w = features.size()
    # features = features.view(b, c, -1)  # [B, C, HW]
    features = features.permute(0,2,3,1)
    class_means = torch.zeros([num_class, c]).to(features.device)
    class_counts = torch.zeros([num_class]).to(features.device)
    class_vars = torch.zeros([num_class, c]).to(features.device)
    for cls in range(num_class):
        ind = (mask == cls)
        # selected_features = bank.get_all(cls).squeeze(0).to(features.device)
        selected_features = features[ind]  # N C
        cur_mean = torch.mean(selected_features, dim=0)
        cur_count = selected_features.size(0)
        diff = selected_features - cur_mean.unsqueeze(0)
        variance = torch.mean(diff ** 2, dim=0)
        # step 5: 存储
        class_means[cls] = cur_mean
        class_counts[cls] = cur_count
        class_vars[cls] = variance
    return class_means, class_counts, class_vars

def find_uncertain_positions(features, proto_means, proto_vars, temp=2):
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
    # posterior = torch.argmax(posterior,dim=1,keepdim=False).reshape(B,H,W)

    # 获取每个位置的最大和次大概率值
    top2_probs, top2_classes = torch.topk(posterior, k=2, dim=1)  # [B*H*W, 2]
    #
    # # 计算最大与次大概率的差
    prob_diff = top2_probs[:, 0] - top2_probs[:, 1]  # [B*H*W]
    # # print(top2_probs[:,0][top2_classes[:,0]!=0]-top2_probs[:,1][top2_classes[:,0]!=0],len(top2_classes[:,0][top2_classes[:,0]!=0]))
    # # 找出差值小于阈值的位置
    # threshold = 0.9
    # uncertain_mask_flat = prob_diff < threshold  # [B*H*W]
    #
    # # 重塑回原始形状 [5,224,224]
    # uncertain_mask = uncertain_mask_flat.reshape(B, H, W)
    prob_diff = prob_diff.reshape(B,H,W)
    posterior = torch.argmax(posterior,dim=1,keepdim=False).reshape(B,H,W)
    return posterior,1-prob_diff


def make_ClassEntropy(tensor, pseudo_label_onehot):
    b, c, h, w = tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)
    tensor_psu = tensor * pseudo_label_onehot
    entropy_per_class = -tensor_psu * torch.log(tensor_psu + 1e-9)
    entropy_per_class = torch.where(entropy_per_class == 0, torch.tensor(0.5), entropy_per_class)
    kernel_size = 32
    patch_size = kernel_size * kernel_size
    entropy_patches = F.unfold(entropy_per_class, kernel_size, stride=kernel_size)  # (24,4,1024,47)
    tensor_patches = F.unfold(tensor, kernel_size, stride=kernel_size)  # (24,4,1024,47)
    entropy_patches = entropy_patches.view(b, c, patch_size, -1)
    tensor_patches = tensor_patches.view(b, c, patch_size, -1)  # (24,4,1024,47)
    mean_entropy_per_patch = entropy_patches.mean(dim=2)  # (24,4,47)
    # print(mean_entropy_per_patch[0])
    min_entropy_patch_indices = torch.argmin(mean_entropy_per_patch, dim=2)  # (24,4)

    # min_entropy_patch_indices_expanded = min_entropy_patch_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, patch_size,1)
    result = torch.zeros(b, c, c, patch_size).cuda()  # (24,4,4, 1024)
    for batch_id in range(b):
        for num_id in range(c):
            patch_idx = min_entropy_patch_indices[batch_id][num_id]
            result[batch_id][num_id] = tensor_patches[batch_id, :, :, patch_idx]
    # min_entropy_patches = torch.gather(tensor_patches,3, min_entropy_patch_indices_expanded)  # (24, 4, 1024)

    return result
def vis_save(original_img, pred, save_path):
    blue = [30, 144, 255]  # aorta
    green = [0, 255, 0]  # gallbladder
    red = [255, 0, 0]  # left kidney
    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    original_img = np.where(pred == 1, np.full_like(original_img, blue), original_img)
    original_img = np.where(pred == 2, np.full_like(original_img, green), original_img)
    original_img = np.where(pred == 3, np.full_like(original_img, red), original_img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img)

def avg_pool2d(input_array, kernel_size, stride):
    N, C, H, W = input_array.shape  # 获取输入数组的形状
    output_h = (H - kernel_size) // stride + 1  # 输出高度
    output_w = (W - kernel_size) // stride + 1  # 输出宽度

    # 初始化输出数组
    output_array = np.zeros((N, C, output_h, output_w))

    for n in range(N):
        for c in range(C):
            for i in range(output_h):
                for j in range(output_w):
                    # 计算池化区域的平均值
                    h_start = i * stride
                    h_end = h_start + kernel_size
                    w_start = j * stride
                    w_end = w_start + kernel_size

                    output_array[n, c, i, j] = np.mean(input_array[n, c, h_start:h_end, w_start:w_end])

    return output_array
def vis_patches(original_img,save_path):
    color_map = {
        1: [0, 0, 255],    # 蓝色
        2: [0, 255, 0],    # 绿色
        3: [255, 0, 0]     # 红色
    }
    image_array = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        image_array[original_img == label] = color
    image = Image.fromarray(image_array)
    image.save(save_path)

def unreliable_map(posterior_probs, softmax_probs):
    """
    margin_confidence: [B, C, H, W]
    posterior_probs:   [B, C, H, W]
    softmax_probs:     [B, C, H, W]
    features:          [B, C_feat, H, W]
    class_centers:     [num_classes, C_feat]
    """
    B, C, H, W = softmax_probs.shape

    # 1. 获取 softmax 和 posterior 的 argmax 类别
    softmax_label = torch.argmax(softmax_probs, dim=1)   # [B, H, W]
    posterior_label = posterior_probs # [B, H, W]

    # 2. 构造不可靠区域掩码
    unreliable_mask = (softmax_label != posterior_label).float()  # [B, H, W]


    return unreliable_mask


import numpy as np
import cv2
import os

import numpy as np
import cv2

def vis_save_entropy(original_img, pred, save_path):
    """
    original_img: 2D numpy array, grayscale image, [H, W], float32 or float64, range [0, 1]
    pred: 2D numpy array, integer label map, [H, W], values in {0, 1, 2, 3}
    """

    # 定义颜色（BGR格式）: 颜色淡一些
    colors = {
        1: (128, 128, 128),     # aorta - red
        2: (0, 255, 0),     # gallbladder - green
        3: (0, 0, 255),     # left kidney - blue
    }

    # 转为 uint8 图像
    img = (original_img * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 创建一个透明图层
    overlay = img_color.copy()

    for cls_id, color in colors.items():
        mask = (pred == cls_id).astype(np.uint8)

        # 轮廓边缘（可选，增强结构感）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=1)

        # 半透明填色
        colored_mask = np.zeros_like(img_color, dtype=np.uint8)
        colored_mask[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.25, 0)

    # 保存图像
    result = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)  # 如果用 matplotlib 显示则保留 RGB
    cv2.imwrite(save_path, result)

def vis_save_image(original_img, save_path):
    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img)
def make_entropy(tensor):
    b, c, h, w = tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)
    # prob_tensor = F.softmax(tensor,dim=1)
    prob_tensor = tensor
    entropy = -torch.sum(prob_tensor * torch.log2(prob_tensor + 1e-10), dim=1)

    return entropy

def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result
def make_onehot(tensor,num_class):
    b, h, w = tensor.size(0),tensor.size(1), tensor.size(2)
    label_flattened = tensor.view(b,-1)
    one_hot_labels = F.one_hot(label_flattened,num_classes=num_class)
    one_hot_labels = one_hot_labels.view(b,h,w,num_class).permute(0,3,1,2)
    return one_hot_labels

def plot_uncertainty_heatmap(image, uncertainty, index=0, cmap='JET', alpha=0.5):
    """
    在原图上叠加不确定性的热力图。

    参数：
    - image: Tensor of shape [B, 1, H, W]
    - uncertainty: Tensor of shape [B, H, W]
    - index: 可视化第 index 个样本
    - cmap: 热力图颜色映射
    - alpha: 热力图透明度
    """
    img = image[index, 0].cpu().numpy()  # [H, W]
    unc = uncertainty[index].cpu().numpy()  # [H, W]

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.imshow(unc, cmap=cmap, alpha=alpha)  # 叠加热力图
    plt.axis('off')
    plt.title("Uncertainty Heatmap Overlay")
    plt.show()
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=[10, 1, 1])
        asd = metric.binary.asd(pred, gt, voxelspacing=[10, 1, 1])
        jac = metric.binary.jc(pred,gt)
        return dice, hd95, asd, jac
    else:
        return 0, 50, 10., 0

def calculate_metric_percase1(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    ratios = []
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    prediction_softmax = np.zeros((label.shape[0],16,label.shape[1],label.shape[2]))
    unreliable_pre = np.zeros_like(image)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        P = zoom(label[ind,:,:], (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        P = torch.from_numpy(P).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urpc":
                out_main, _, _, _ = net(input)
            else:
                out_main,features= net(input)
            outputs_weak_logits = out_main
            outputs_weak_soft2 = torch.softmax(out_main,dim=1)
            pseudo_mask2 = (normalize(outputs_weak_soft2) > 0.95)
            pseudo_mask2 = pseudo_mask2 * outputs_weak_soft2
            pseudo_mask2 = torch.argmax(pseudo_mask2,dim=1).squeeze(0).cpu().detach().numpy()
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            means,counts,vars = make_prototype_center(features,P.squeeze(0),4)
            post,diff = find_uncertain_positions(features,means,vars)
            # B,H,W = out.unsqueeze(0).shape
            # mask_entropy = ~unreliable_map(post,outputs_weak_soft2).bool()
            # mask_entropy = mc_dropout_uncertainty_entropy(net,input).cpu()
            mask_entropy = make_entropy(outputs_weak_soft2).cpu()
            reliable_thresh = mask_entropy < 0.75
            ratio = confident_incorrect_ratio(out.unsqueeze(0).cpu(), P.squeeze(0).cpu(), reliable_thresh.squeeze(0).cpu())
            out = out.cpu().detach().numpy()
            ratios.append(ratio)
            outputs_weak_soft2 = outputs_weak_soft2.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            # position = (reliable_thresh != 0)
            # position = zoom(position.squeeze(0),(x / 224, y / 224), order=0)
            unreliable_pixel = zoom(mask_entropy.detach().cpu().numpy().squeeze(0),(x / 224, y / 224), order=0)
            # print(mask_entropy.shape,input.shape)
            # unreliable_pre[ind] = unreliable_pixel * pred
            prediction[ind] = pred
            unreliable_pre[ind] = unreliable_pixel

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    image_path = os.path.join(test_save_path,'result/')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    for i in range(image.shape[0]):
        # vis_save(image[i],prediction[i],image_path + case + "_pre_{}.png".format(i))
        # vis_save(image[i],label[i],image_path + case + "_gt_{}.png".format(i))
        save_uncertainty_overlay(image[i], unreliable_pre[i],
                                 image_path + case + "_uncertainty_{}.png".format(i))
        # vis_save_entropy(image[i],unreliable_pre[i],image_path + case + "_entropymap_{}.png".format(i))
        # vis_save_image(image[i],image_path + case + "_input_{}.png".format(i))
    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # unre_itk = sitk.GetImageFromArray(unreliable_pre.astype(np.float32))
    # unre_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    # sitk.WriteImage(unre_itk, test_save_path + case + "_unre.nii.gz")
    return first_metric, second_metric, third_metric,  np.mean(ratios)


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "result/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "reulst/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)

    save_mode_path = os.path.join(
        snapshot_path, 'E:\workspace\Semi-Supervised\model\My_CPS_3_labeled/unet_F\model1_iter_4000_dice_0.6007.pth'.format(FLAGS.model))
    resume = torch.load(save_mode_path)
    new = net.state_dict()
    pretrained_dict = {}
    for k, v in resume['state_dict'].items():
        for kk in new.keys():
            if kk in k:
                pretrained_dict[kk] = v
                break
    new.update(pretrained_dict)
    net.load_state_dict(new)
    # proj = net_factory(net_type='projector', in_chns=4,
    #                   class_num=FLAGS.num_classes)
    # save_mode_path = os.path.join(
    #     snapshot_path, 'D:\workspace\Semi-Supervised\model\ACDC\Cross_Pseudo_Supervision_7_labeled/unet/unet_best_model2.pth'.format(FLAGS.model))
    # resume = torch.load(save_mode_path)
    # new = proj.state_dict()
    # pretrained_dict = {}
    # for k, v in resume['state_dict'].items():
    #     for kk in new.keys():
    #         if kk in k:
    #             pretrained_dict[kk] = v
    #             break
    # new.update(pretrained_dict)
    # proj.load_state_dict(new)
    # net.load_state_dict(torch.load(save_mode_path))
    # resume = torch.load('D:\workspace\Semi-Supervised\model\ACDC\My_model_Finalversion_14_labeled/unet/unet_best_projection5.pth')
    # new = proj.state_dict()
    # pretrained_dict = {}
    # for k, v in resume['state_dict'].items():
    #     for kk in new.keys():
    #         if kk in k:
    #             pretrained_dict[kk] = v
    #             break
    # new.update(pretrained_dict)
    # proj.load_state_dict(new)
    print("init weight from {}".format(save_mode_path))
    print("save image to {}".format(test_save_path))
    net.eval()
    ratios = []
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric,ratio = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        ratios.append(ratio)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    print(np.mean(ratios))
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)