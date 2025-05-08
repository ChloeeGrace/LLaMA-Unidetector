# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import torch.nn as nn
from util.utils import slprint, to_device
from torch.cuda.amp import autocast
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from scipy.optimize import linear_sum_assignment


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
             args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_fb_di'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #
        #     f0 = open(od_dst, "w")
        #     for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, boxes):
        #         f0.write("%s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(xmin), float(ymin), float(xmax), float(ymax)))
        #     f0.close()

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_fb'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax) in zip(scores_fb, scores, labels, boxes):
        #         f0.write("%s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax)))
        #     f0.close()

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_allclass'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['score_all']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10), float(s11), float(s12), float(s13), float(s14), float(s15)))
        #     f0.close()

        # dota1.0
        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_entire'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['score_all']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10), float(s11), float(s12), float(s13), float(s14), float(s15)))
        #     f0.close()

        # dior
        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dior_txt_entire'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['score_all']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10), float(s11), float(s12), float(s13), float(s14), float(s15), float(s16), float(s17), float(s18), float(s19), float(s20)))
        #     f0.close()

        # dotadior
        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dotadior_txt_entire'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['score_all']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (
        #     s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15) in zip(
        #             scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax),
        #             float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9),
        #             float(s10), float(s11), float(s12), float(s13), float(s14), float(s15)))
        #     f0.close()

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_fb_mul_di'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax) in zip(scores_fb, scores, labels, boxes):
        #         sc = score_fb * score
        #         f0.write("%s %s %s %s %s %s\n" % (
        #             int(label), float(sc), float(xmin), float(ymin), float(xmax), float(ymax)))
        #     f0.close()

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_fb_f1socre'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax) in zip(scores_fb, scores, labels, boxes):
        #         f1_sc = 2*score_fb * score/(score_fb + score)
        #         f0.write("%s %s %s %s %s %s\n" % (
        #             int(label), float(f1_sc), float(xmin), float(ymin), float(xmax), float(ymax)))
        #     f0.close()

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


def enable_dropout(model):
    """启用推理阶段的dropout"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # 将dropout设为train模式，即推理时启用


# 将 (x0, y0, w, h) 转换为 (x0, y0, x1, y1)
def convert_boxes_format(boxes):
    """
    将 (x0, y0, w, h) 格式转换为 (x0, y0, x1, y1)
    :param boxes: [N, 4] 的 bbox 张量，格式为 (x0, y0, w, h)
    :return: [N, 4] 的 bbox 张量，格式为 (x0, y0, x1, y1)
    """
    x0, y0, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x0 + w
    y1 = y0 + h
    return torch.stack([x0, y0, x1, y1], dim=-1)


# 计算两个bbox集合的IoU
def compute_iou(boxes1, boxes2):
    """
    计算两个bbox集合的IoU
    :param boxes1: [N, 4]的bbox张量 (x0, y0, x1, y1)
    :param boxes2: [M, 4]的bbox张量 (x0, y0, x1, y1)
    :return: IoU矩阵 [N, M]
    """
    N = boxes1.size(0)
    M = boxes2.size(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 左上角坐标
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下角坐标

    wh = (rb - lt).clamp(min=0)  # 计算宽高
    inter = wh[:, :, 0] * wh[:, :, 1]  # 相交区域面积

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2 - inter  # 联合面积

    iou = inter / union  # 计算IoU
    return iou


def remove_duplicate_boxes(boxes, iou_threshold=0.9):
    iou_matrix = compute_iou(boxes, boxes)
    keep_indices = []

    for i in range(iou_matrix.size(0)):
        # 如果当前框与之前保留的框IoU都小于阈值，则保留该框
        if all(iou_matrix[i, j] < iou_threshold for j in keep_indices):
            keep_indices.append(i)

    return torch.tensor(keep_indices, device=boxes.device)


def match_boxes(predictions, iou_threshold=0.7):
    num_samples = len(predictions)
    all_boxes = []
    all_objectness = []

    # 收集所有推理中的 bbox 和 objectness 分数
    for pred in predictions:
        boxes = pred['pred_boxes'].squeeze(0)
        objectness = torch.sigmoid(pred['pred_logits'].squeeze(0)[:, -1])
        all_boxes.append(boxes)
        all_objectness.append(objectness)

    # 使用第一组预测作为基准
    base_boxes = all_boxes[0]
    base_boxes_converted = convert_boxes_format(base_boxes)

    # 存储每个框被匹配的次数和对应的 objectness
    match_counts = torch.zeros(base_boxes.size(0), dtype=torch.int32, device=base_boxes.device)
    all_matched_objectness = torch.zeros_like(base_boxes[:, 0], device=base_boxes.device)

    # 遍历其他推理结果，进行IoU匹配
    for i in range(1, num_samples):
        current_boxes = all_boxes[i]
        current_boxes_converted = convert_boxes_format(current_boxes)

        # 计算 IoU 矩阵
        iou_matrix = compute_iou(base_boxes_converted, current_boxes_converted)

        # 使用匈牙利算法进行最佳匹配
        row_indices, col_indices = linear_sum_assignment(-iou_matrix.cpu().numpy())

        # 过滤掉 IoU 低于阈值的匹配，并统计匹配次数
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= iou_threshold:
                match_counts[row] += 1  # 增加匹配次数
                all_matched_objectness[row] += all_objectness[i][col]  # 累加匹配的 objectness

    return match_counts, all_matched_objectness, base_boxes, all_boxes


# 根据检测次数修正 objectness 和 bbox
def adjust_objectness_and_fuse_bboxes(match_counts, matched_objectness, base_boxes, all_boxes, alpha=0.7,
                                      min_detection_count=3):
    """
    根据检测次数修正 objectness，并融合 bbox
    :param match_counts: 每个框的检测次数
    :param matched_objectness: 累积的 objectness 分数
    :param base_boxes: 基准框
    :param all_boxes: 所有推理的 bbox
    :param alpha: 平衡系数，控制检测次数对 objectness 修正的影响
    :param min_detection_count: 最小检测次数，低于此次数的框会被降低 objectness
    :return: 修正后的 bbox 和 objectness
    """
    num_boxes = base_boxes.size(0)

    # 过滤掉匹配次数为 0 的框
    valid_indices = (match_counts > 0).nonzero(as_tuple=True)[0]

    # 删除匹配次数为 0 的框
    base_boxes = base_boxes[valid_indices]
    match_counts = match_counts[valid_indices]
    matched_objectness = matched_objectness[valid_indices]

    # 计算最终的 objectness 分数
    final_objectness = matched_objectness / match_counts.clamp(min=1)  # 计算均值
    final_objectness[match_counts < min_detection_count] *= (1 - alpha)  # 如果检测次数低于阈值，降低 objectness

    # 对 bbox 进行融合
    fused_bboxes = torch.zeros_like(base_boxes)
    for i in range(base_boxes.size(0)):
        if match_counts[i] >= min_detection_count:
            # 融合匹配的 bbox 均值
            matched_boxes = torch.stack([all_boxes[j][valid_indices[i]] for j in range(len(all_boxes))])
            fused_bboxes[i] = matched_boxes.mean(dim=0)  # 使用均值进行融合
        else:
            # 如果检测次数不够，保留原始框
            fused_bboxes[i] = base_boxes[i]

    return fused_bboxes, final_objectness


# 执行主流程
def monte_carlo_adjustment(predictions, iou_threshold=0.7, alpha=0.7, min_detection_count=3):
    match_counts, matched_objectness, base_boxes, all_boxes = match_boxes(predictions, iou_threshold=iou_threshold)
    fused_bboxes, final_objectness = adjust_objectness_and_fuse_bboxes(match_counts, matched_objectness, base_boxes,
                                                                       all_boxes, alpha=alpha,
                                                                       min_detection_count=min_detection_count)

    # 创建新的 prediction，将调整后的 bbox 和 objectness 替换原来的
    final_predictions = predictions[0].copy()  # 使用第一个推理结果作为模板
    # 过滤掉未匹配的框
    valid_indices = (match_counts > 0).nonzero(as_tuple=True)[0]

    final_predictions['pred_boxes'] = fused_bboxes.unsqueeze(0)
    final_logits = final_predictions['pred_logits'][:, valid_indices, :].clone()
    final_objectness_log = torch.log(final_objectness / (1 - final_objectness + 1e-6))
    final_logits[:, :, -1] = final_objectness_log.unsqueeze(0)
    final_predictions['pred_logits'] = final_logits

    return final_predictions


@torch.no_grad()
def evaluate_txt(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
                 args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    enable_dropout(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:

                predictions = []

                for _ in range(5):
                    with torch.no_grad():
                        # 前向传播获取一次预测结果
                        output = model(samples)
                        predictions.append(output)
            else:
                outputs = model(samples)

        # outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])
        outputs = monte_carlo_adjustment(predictions)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        od_source = '/data/XJL/dino/DINO/out/test/txt/out_RSdet_object_dior_txt_mtkl'
        if not os.path.exists(od_source):
            os.mkdir(od_source)
        for k, v in res.items():
            file = str(k) + '.txt'
            od_dst = os.path.join(od_source, file)
            scores = v['scores']
            labels = v['labels']
            boxes = v['boxes']

            f0 = open(od_dst, "w")
            for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, boxes):
                f0.write("%s %s %s %s %s %s\n" % (
                    int(label), float(score), float(xmin), float(ymin), float(xmax), float(ymax)))
            f0.close()

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_fb'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax) in zip(scores_fb, scores, labels, boxes):
        #         f0.write("%s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax)))
        #     f0.close()

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_allclass'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['score_all']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10), float(s11), float(s12), float(s13), float(s14), float(s15)))
        #     f0.close()

        # dota1.0
        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_entire'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['score_all']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10), float(s11), float(s12), float(s13), float(s14), float(s15)))
        #     f0.close()

        # dior
        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dior_txt_entire'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['score_all']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10), float(s11), float(s12), float(s13), float(s14), float(s15), float(s16), float(s17), float(s18), float(s19), float(s20)))
        #     f0.close()

        # # dotadiorfb 111111111111111
        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dotadior_txt_entire'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['score_all']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (
        #             s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15) in zip(
        #         scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax),
        #             float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9),
        #             float(s10), float(s11), float(s12), float(s13), float(s14), float(s15)))
        #     f0.close()

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_fb_mul_di'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax) in zip(scores_fb, scores, labels, boxes):
        #         sc = score_fb * score
        #         f0.write("%s %s %s %s %s %s\n" % (
        #             int(label), float(sc), float(xmin), float(ymin), float(xmax), float(ymax)))
        #     f0.close()

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_dota1.0_txt_fb_f1socre'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax) in zip(scores_fb, scores, labels, boxes):
        #         f1_sc = 2*score_fb * score/(score_fb + score)
        #         f0.write("%s %s %s %s %s %s\n" % (
        #             int(label), float(f1_sc), float(xmin), float(ymin), float(xmax), float(ymax)))
        #     f0.close()


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None,
         logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                    "image_id": int(image_id),
                    "category_id": l,
                    "bbox": b,
                    "score": s,
                }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)

    return final_res
