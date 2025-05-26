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


                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()
                logging.info(
                    'current best dice coef model 1 {}, model 2 {}'.format(best_performance1, best_performance2))

            
            if iter_num >= max_iterations:
                break
        r_loss_seg_dice_mean.append(r_loss_seg_dice_total / len(trainloader))
        v_loss_seg_dice_mean.append(v_loss_seg_dice_total / len(trainloader))

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
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    snapshot_path = "model/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
