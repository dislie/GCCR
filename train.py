# -*- "coding: utf-8" -*-

from datetime import datetime
import math
import time
import os
import logging
import argparse
from contextlib import suppress
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import numpy as np
import matplotlib
from torch.utils.tensorboard import SummaryWriter

from timm.utils import NativeScaler

matplotlib.use("agg")

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

from timm.layers import resample_abs_pos_embed
from timm.utils.summary import update_summary
from collections import OrderedDict

from thop import profile
from network.GlobalBranch import GCCR, create_global_branch
from modules.datasets import BatchDataset, BalancedBatchSampler
from modules import utils, losses

utils.fix_seed()


def main():
    # prepare model
    load_pretrained = True

    if args.img_size == 224:
        size = 224
    batch_size = (args.sample_classes * args.sample_images)
    model_cfg = {
        "num_classes": cfg.num_classes,
        "image_size": args.img_size,
        "load_pretrained": load_pretrained,
        "drop_rate": cfg.drop_rate,
        "window_size": args.window_size,
        "keep_rate": cfg.keep_rate,
        "pruning_loc": cfg.pruning_loc,
        'pretrain_path': cfg.pretrain_path,

        'nodynamic': args.nodynamic,

    }
    local_cfg = {
        "depth": args.local_depth,
        "batch_size": batch_size,
    }
    model = GCCR(model_cfg, local_cfg, args.NoBatch_GNN, args.NoAWF,args.NoRank_loss)

    logging.info("Calculate MACs & FLOPs ...")
    inputs = torch.randn((1, 3, args.img_size, args.img_size))
    macs, num_params = profile(model, (inputs,), verbose=False)  # type: ignore
    logging.info(
        "\nParams(M):{:.2f}, MACs(G):{:.2f}, FLOPs(G):~{:.2f}".format(num_params / (1000 ** 2), macs / (1000 ** 3),
                                                                      2 * macs / (1000 ** 3)))
    logging.info("")

    if not args.nodistill:
        teacher_model: nn.Module = create_global_branch(model_cfg,
                                                        only_teacher_model=True)  # type: ignore
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model = None
    logging.info("\nargs: \n{}".format(args))
    logging.info("\nconfigs: \n{}".format(cfg))
    logging.info("\nmodel: \n{}".format(model))

    ### tensorboard
    tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # load trained weights to model
    if args.val_dir:

        weights_path = os.path.join(args.val_dir, "best.pth")
        logging.info("Load weights from {}".format(weights_path))
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    ####
    model.to(device)

    scale_size = int(round(512 * args.img_size / 448))
    transform1 = transforms.Compose([
        transforms.Resize([scale_size, scale_size]),
        transforms.RandomCrop([args.img_size, args.img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )])
    transform2 = transforms.Compose([
        transforms.Resize([scale_size, scale_size]),
        transforms.CenterCrop([args.img_size, args.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )])

    [backbone_params, other_params] = model.get_param_groups()

    ### optimizers, loss functions
    optimizers = [
        torch.optim.AdamW(backbone_params, lr=cfg.backbone_lr, weight_decay=cfg.weight_decay, betas=cfg.betas),
        torch.optim.AdamW(other_params, lr=cfg.others_lr, weight_decay=cfg.weight_decay, betas=cfg.betas),
    ]

    schedulers = [
        utils.WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_epochs, t_total=int(1.1 * args.epochs))
        for optimizer in optimizers
    ]
    ce_criterion = losses.LabelSmoothingCrossEntropy().to(device)
    rank_criterion = None if args.NoRank_loss else nn.MarginRankingLoss(margin=0.05).to(device)

    ###
    if args.nodynamic:
        dynamic_criterion = None
    else:
        if args.ratio_weight is None:
            args.ratio_weight = 10
        dynamic_criterion = losses.ConvNextDistillDiffPruningLoss(teacher_model, ratio_weight=args.ratio_weight,
                                                                  distill_weight=0.5, keep_ratio=model.keep_rate,
                                                                  swin_token=True)

    ### resume training
    start_epoch = 0
    best_val = None
    # if args.weights_dir and not args.finetune:
    #     state_dict = torch.load(os.path.join(args.weights_dir, "params.pth"), map_location="cpu")
    #     start_epoch = state_dict["epoch"]
    #     [optimizers[idx].load_state_dict(dict) for idx,dict in enumerate(state_dict['optimizer_state_dicts'])]
    #     best_val = state_dict["best_val"]

    # Data loading
    train_dataset = BatchDataset(cfg.data_path, 'train', cfg.txt_dir, transform=transform1)
    if args.NoRank_loss:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_sampler = BalancedBatchSampler(train_dataset, args.sample_classes, args.sample_images)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                                   num_workers=cfg.num_workers, pin_memory=True)

    val_dataset = BatchDataset(cfg.data_path, 'val', cfg.txt_dir, transform=transform2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)

    start_time = datetime.now().replace(microsecond=0)
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []

    best_epoch = 0


    if args.val_dir:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4,
                                                 pin_memory=True)
        val_loss, val_acc = validate(val_loader, model, ce_criterion, 0)
        print(f"val_loss: {val_loss:.6f}, val_acc: {val_acc:.6f}")
        exit(0)


    # amp
    args.use_amp = cfg.use_amp
    amp_autocast = suppress  # do nothing
    Grad_scaler = None
    if args.use_amp:
        #
        Grad_scaler = GradScaler()
        amp_autocast = autocast

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        loss, acc, loss_detail= train(train_loader, model, [ce_criterion, rank_criterion, dynamic_criterion],
                                       optimizers, epoch,
                                       amp_autocast, Grad_scaler)
        [scheduler.step() for scheduler in schedulers]
        loss_list.append(loss)
        acc_list.append(acc)
        # validate
        val_loss, val_acc = validate(val_loader, model, ce_criterion, epoch)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        end = time.time()
        logging.info(
            "[Epoch:{}/{}] using_time:{:.2f}s lr1:{:.7f} lr2:{:.7f} loss:{:.6f} acc:{:.6f} val_loss:{:.6f} val_acc:{:.6f}".format(
                epoch + 1, args.epochs, end - start, optimizers[0].param_groups[0]['lr'],
                optimizers[1].param_groups[0]['lr'], loss, acc, val_loss, val_acc
            )
        )

        utils.plot_history(loss_list, acc_list, val_loss_list, val_acc_list, history_save_path)
        # save model
        torch.save({
            'epoch': epoch + 1,
            'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
            'best_val': best_val,
        }, params_save_path)
        torch.save(model.state_dict(), os.path.join(model_last_path))
        # update summary
        train_metrics = OrderedDict([('loss', loss), ('acc', acc)])
        eval_metrics = OrderedDict([('loss', val_loss), ('acc', val_acc)])
        update_summary(
            epoch + 1, train_metrics, eval_metrics,
            os.path.join(cfg.output_dir, '{}-{}/summary.csv'.format(args.name, timestamp)),
            write_header=best_val is None)

        #
        tb_writer.add_scalar('Loss/train', loss, epoch)
        tb_writer.add_scalar('Accuracy/train', acc, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
        if loss_detail:
            for k, v in loss_detail.items():
                tb_writer.add_scalar(f'{k}/train', v, epoch)

        if best_val is None or val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_best_path)
            logging.info("Saved best model.")

    utils.plot_history(loss_list, acc_list, val_loss_list, val_acc_list, history_save_path)
    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info('Training time: {} days {:.2f} hours'.format((
        datetime.now().replace(microsecond=0) - start_time).days,
        round((datetime.now().replace(microsecond=0) - start_time).seconds / 3600, 2)))
    logging.info('Best val acc: {:.4f} at epoch: {}'.format(best_val, best_epoch))



def cal_loss(model, inputs, targets, criterions, amp_autocast, with_acc=False):
    ce_criterion, rank_criterion, dynamic_criterion = criterions
    acc = 0

    if args.NoRank_loss:
        with amp_autocast():
            logits, global_feature, decision_mask_list = model(inputs)
    else:
        with amp_autocast():
            logits, self_scores, other_scores, global_feature, decision_mask_list= model(
                inputs, targets)
    with amp_autocast():
        loss = 2 * ce_criterion(logits, targets)  # softmax_loss
    loss_dict = OrderedDict()
    loss_dict.update({'ce_loss': loss.item()})
    loss_str = f"ce_loss:{loss.item():.6f}, "

    if not args.NoRank_loss:
        flags = torch.ones([self_scores.size(0), ]).to(device)
        with amp_autocast():
            rank_loss = rank_criterion(self_scores, other_scores, flags)  # rank_loss
        loss +=  rank_loss
        loss_str += f"rank_loss:{rank_loss.item():.6f}, "
        loss_dict.update({'rank_loss': rank_loss.item()})

    if dynamic_criterion:
        with amp_autocast():
            dynamic_loss = dynamic_criterion(inputs, [global_feature, decision_mask_list])
        loss +=  dynamic_loss
        loss_str += f"dynamic_loss:{dynamic_loss.item():.6f}"
        loss_dict.update({'dynamic_loss': dynamic_loss.item()})

    if with_acc:
        acc = utils.cal_accuracy(logits, targets)

    if torch.isnan(loss):
        logging.error("Nan is detected in total loss!")
        exit(-1)

    if with_acc:
        return loss, loss_dict, acc
    else:
        return loss, loss_dict


def loss_dict_avg(loss_dict_list):
    avg_loss_dict = {}
    for dict in loss_dict_list:
        for k, v in dict.items():
            if k not in avg_loss_dict:
                avg_loss_dict[k] = [v]
            else:
                avg_loss_dict[k].append(v)
    for k, v in avg_loss_dict.items():
        avg_loss_dict[k] = np.mean(v)
    return avg_loss_dict


def train(train_loader, model, criterions, optimizers, epoch, amp_autocast, Grad_scaler):
    model.train()
    batch_loss_list = []
    batch_acc_list = []
    loss_dict_list = []

    total = len(train_loader)
    for i, (inputs, targets, filenames) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        [optimizer.zero_grad() for optimizer in optimizers]

        loss, loss_str, acc= cal_loss(model, inputs, targets, criterions, amp_autocast,
                                       with_acc=True)

        if Grad_scaler:
            Grad_scaler.scale(loss).backward()
            # Backbone
            Grad_scaler.step(optimizers[0])
            #
            Grad_scaler.step(optimizers[1])

            Grad_scaler.update()
        else:
            loss.backward()
            [optimizer.step() for optimizer in optimizers]

        loss_dict_list.append(loss_str)
        if i % cfg.log_step == 0:
            logging.info(
                "Training epoch:{}/{} batch:{}/{} loss:{:.6f} acc:{:.6f} loss_detail: {}".format(epoch + 1, args.epochs,
                                                                                                 i + 1, total,
                                                                                                 loss.item(), acc,
                                                                                                 loss_dict_avg(
                                                                                                     loss_dict_list)))
        batch_loss_list.append(loss.item())
        batch_acc_list.append(acc)

    loss_str = loss_dict_avg(loss_dict_list)
    return np.mean(batch_loss_list), np.mean(batch_acc_list), loss_str

def validate(val_loader, model, ce_criterion, epoch):
    model.eval()  # switch to evaluate mode
    batch_loss_list = []
    batch_acc_list = []
    with torch.no_grad():
        total = len(val_loader)
        for i, (inputs, targets, filenames) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)  #
            if len(outputs.size()) == 1:
                outputs = torch.unsqueeze(outputs, dim=0)

            loss = ce_criterion(outputs, targets)
            batch_loss_list.append(loss.item())

            acc = utils.cal_accuracy(outputs, targets)
            batch_acc_list.append(acc)

            if i % 50 == 0:
                logging.info(
                    "Validating epoch:{}/{} batch:{}/{} loss:{:.6f} acc:{:.6f}".format(epoch + 1, args.epochs, i + 1,
                                                                                       total, loss.item(), acc))
    return np.mean(batch_loss_list), np.mean(batch_acc_list)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--sample_classes", type=int, default=2, help="sample n classes from all classes each time")
    parser.add_argument("--sample_images", type=int, default=10, help="sample n images from each classes each time")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--window_size", type=int, default=7,
                        help="image_size:224,window_size:7, image_size:384,window_size:12;")

    parser.add_argument("--val_dir", type=str, default=None, help=".pth weights directory")

    parser.add_argument("--nodistill", action="store_true", help=" without using teacher model")
    parser.add_argument("--nodynamic", action="store_true", help="without dynamic design in global branch")

    parser.add_argument("--NoBatch_GNN", action="store_true")
    parser.add_argument("--NoAWF", action="store_true")
    parser.add_argument("--NoRank_loss", action="store_true")

    parser.add_argument("--vis_mode", action="store_true", help="only visualize")

    parser.add_argument("--ratio_weight", type=int, default=None, help="if None, set 2 for vit, set 10 for swin.")

    parser.add_argument("--local_from_stage", type=int, default=-1,
                        help="[0,1,2,3] for SwinT or -1 for all architecture")
    parser.add_argument("--local_depth", type=int, default=1, help="number of blocks in BatchBranch")

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

    model_best_path = "./{}/{}-{}/models/best.pth".format(cfg.output_dir, args.name, timestamp)
    model_last_path = "./{}/{}-{}/models/last.pth".format(cfg.output_dir, args.name, timestamp)
    params_save_path = "./{}/{}-{}/models/params.pth".format(cfg.output_dir, args.name, timestamp)
    log_path = "./{}/{}-{}/out_logs/out.log".format(cfg.output_dir, args.name, timestamp)
    history_save_path = "./{}/{}-{}/history/history.png".format(cfg.output_dir, args.name, timestamp)

    tensorboard_log_dir = "./{}/{}-{}/tensorboard/".format(cfg.output_dir, args.name, timestamp)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    os.makedirs("./{}/{}-{}/out_logs/".format(cfg.output_dir, args.name, timestamp), exist_ok=True)
    os.makedirs("./{}/{}-{}/models/".format(cfg.output_dir, args.name, timestamp), exist_ok=True)
    os.makedirs("./{}/{}-{}/history/".format(cfg.output_dir, args.name, timestamp), exist_ok=True)

    logging.basicConfig(
        level="INFO",
        handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()]
    )

    part_attn_save_dir = "./{}/{}-{}/parts-attn/".format(cfg.output_dir, args.name, timestamp)
    os.makedirs("./{}/{}-{}/parts-attn/".format(cfg.output_dir, args.name, timestamp), exist_ok=True)

    if torch.cuda.is_available() and args.gpus != "cpu":
        device = torch.device(f'cuda:{args.gpus}')
    else:
        device = torch.device("cpu")

    main()
