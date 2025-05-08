# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


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


@torch.no_grad()
def evaluate_txt(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
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

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # od_source = '/data/XJL/open-vocabulary/DINO-main/out/test/txt/regular_fb_dior_txt1'
        od_source = '/data/XJL/open-vocabulary/DINO-main/out/test/txt/RSdet_objectsmall_dota_txttttest'
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
        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_RSdet_objectsmall_debug_dior_txt'
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

        # od_source = '/data/XJL/dino/DINO/out/test/txt/out_RSdet_objectsmall_dota_txt'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']

        #     f0 = open(od_dst, "w")
        #     for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, boxes):
        #         f0.write("%s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(xmin), float(ymin), float(xmax), float(ymax)))
        #     f0.close()

        # od_source = '/data/XJL/open-vocabulary/DINO-main/out/test/txt/out_RSdet_objectsmall_nwpu_txt'  # real
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k, v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']

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

        # nwpu 21classes
        # od_source = '/data/XJL/open-vocabulary/DINO-main/out/test/txt/out_RSdet_objectsmall_finetune_nwpu_txt'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['scores_all']
        
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10)))
        #     f0.close()
        
        # nwpu 11classes
        # od_source = '/data/XJL/open-vocabulary/DINO-main/out/test/txt/out_RSdet_objectsmall_finetune_nwpu_txt'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['scores_all']
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10)))
        #     f0.close()
        
        # dota1.0 16classes
        # od_source = '/data/XJL/open-vocabulary/DINO-main/out/test/txt/out_RSdet_objectsmall_finetune_dota_txt16'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['scores_all']
        
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10), float(s11), float(s12), float(s13), float(s14), float(s15)))
        #     f0.close()
        
        # dota1.0 21classes
        # od_source = '/data/XJL/open-vocabulary/DINO-main/out/test/txt/out_RSdet_objectsmall_finetune_dota_txt'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['scores_all']
        
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
        #             int(label), float(score), float(score_fb), float(xmin), float(ymin), float(xmax), float(ymax), float(s1), float(s2), float(s3), float(s4), float(s5), float(s6), float(s7), float(s8), float(s9), float(s10), float(s11), float(s12), float(s13), float(s14), float(s15)))
        #     f0.close()

        # dior
        # od_source = '/data/XJL/open-vocabulary/DINO-main/out/test/txt/out_RSdet_objectsmall_finetune_dior_txt'
        # if not os.path.exists(od_source):
        #     os.mkdir(od_source)
        # for k,v in res.items():
        #     file = str(k) + '.txt'
        #     od_dst = os.path.join(od_source, file)
        #     scores_fb = v['scores_fb']
        #     scores = v['scores']
        #     labels = v['labels']
        #     boxes = v['boxes']
        #     score_all = v['scores_all']
        
        #     f0 = open(od_dst, "w")
        #     for score_fb, score, label, (xmin, ymin, xmax, ymax), (s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20) in zip(scores_fb, scores, labels, boxes, score_all):
        #         f0.write("%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (
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
