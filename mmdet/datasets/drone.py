# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset

from mmcv.ops.nms import nms
import time
from .pipelines import Compose
from collections import defaultdict
import json


@DATASETS.register_module()
class DroneDataset(CocoDataset):
    CLASSES = ('bird',)

    def __init__(self,
                 # copied from custom.py
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),

                 # newly added
                 hard_negative_file=None,
                 # hard_negative_config=None,
                 ):
        # super().__init__(
        #     ann_file=ann_file,
        #     pipeline=pipeline,
        #     classes=classes,
        #     data_root=data_root,
        #     img_prefix=img_prefix,
        #     seg_prefix=seg_prefix,
        #     proposal_file=proposal_file,
        #     test_mode=test_mode,
        #     filter_empty_gt=filter_empty_gt,
        #     file_client_args=file_client_args
        # )

        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)

        self.hard_negative_file = hard_negative_file
        # self.hard_negative_config = hard_negative_config

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
            # -------------
            if not (self.hard_negative_file is None
                    or osp.isabs(self.hard_negative_file)):
                self.hard_negative_file = osp.join(self.data_root,
                                              self.hard_negative_file)

        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        if self.hard_negative_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.hard_negative_file) as local_path:
                    self.hard_negatives = self.load_hard_negatives(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.hard_negative_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.hard_negatives = self.load_hard_negatives(self.hard_negative_file)
        else:
            self.hard_negatives = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def load_hard_negatives(self, hard_negative_file):
        """Load annotation from saved hard negative file.

        Args:
            hard_negative_file (str): Path of hard negative file.

        Returns:
            dict[image_ids]: the value of each key is a list of bounding boxes
        """
        hard_negative_data = json.load(open(hard_negative_file, 'r'))
        assert type(hard_negative_data) == list, 'hard_negative file format {} not supported'.format(type(hard_negative_data))

        hard_negatives = defaultdict(list)
        for det in hard_negative_data:  # a list of dict
            # {"id": 0, "image_id": 0,
            # "bbox": [2941.655029296875, 1440.904052734375, 42.48583984375, 39.647705078125],
            # "area": 1684.4660481214523, "category_id": 0}
            x1, y1, w, h = det['bbox']
            if det['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            hard_negatives[det['image_id']].append(bbox)

        # Create empty list for image_ids without any false positives.
        assert hasattr(self, 'img_ids'), 'COCO annotations should be initialized first before' \
                                         ' loading hard negatives.'
        for i in self.img_ids:
            if i not in hard_negatives:
                hard_negatives[i] = np.zeros((0, 4), dtype=np.float32)
            else:
                hard_negatives[i] = np.array(hard_negatives[i], dtype=np.float32)
        return hard_negatives

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        # -----------------------
        if self.hard_negatives is not None:
            results['hard_negatives'] = self.hard_negatives[idx]  # array, x1, y1, x2, y2
        # -----------------------
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None
                 ):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        # import os
        # from tti.tti_conf import LIB_ROOT_DIR
        # jsonfile_prefix = os.path.join(LIB_ROOT_DIR, 'work_dirs/centernet_resnet18_140e_coco/epoch_20')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        # default iou threshold for tiny object detection.
        if iou_thrs is None:
            iou_thrs = np.array([0.25, 0.5, 0.75])

        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def generate_hard_negatives(self,
                                results,
                                output_path,
                                jsonfile_prefix=None,
                                num_max_det=10,
                                pos_iou_thr=1e-5,  # iou_thr for judging a det as positive
                                score_thd=0.05,
                                nms_thd=0.05,
                                ):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
        :param jsonfile_prefix:
        :param pos_iou_thr:

        """
        print('Generating hard negative examples ...')
        assert output_path is not None

        # if jsonfile_prefix is None:
        #     timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        #     jsonfile_prefix = osp.join(output_dir, f'eval_json_results_{timestamp}.json')
        #     # jsonfile_prefix = osp.join(output_dir, 'json_results')
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if 0 < nms_thd < 1.0:
            for k in range(len(results)):
                # there is always only 1 item in each results[k]
                assert len(results[k]) == 1
                det = results[k][0]  # n, 5
                results[k][0] = nms(boxes=det[:, :4], scores=det[:, 4],
                                    iou_threshold=pos_iou_thr,
                                    )[0]  # return is a tuple

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
        iou_type = 'bbox'
        metric = 'bbox'
        predictions = mmcv.load(result_files[metric])
        coco_det = coco_gt.loadRes(predictions)

        cocoEval = COCOeval(coco_gt, coco_det, iou_type)
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = [num_max_det]
        cocoEval.params.iouThrs = [pos_iou_thr]

        cocoEval.evaluate()
        hard_negative_result = self.identify_hard_negatives(cocoEval, score_thd=score_thd,
                                                            max_det=num_max_det)
        mmcv.dump(hard_negative_result, output_path)
        print('Generating hard negative examples done')

    @staticmethod
    def identify_hard_negatives(cocoEval, score_thd=-float('inf'), max_det=float('inf')):
        """
         cocoEval.evalImgs = [ {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
        }, ..., ]
        """

        count = 0
        result = []
        
        # TODO: check this bug:
        #if eval_img is None or eval_img['aRng'] != p.areaRng[0] or eval_img['maxDet'] != 1000:
        #continue
        for k, eval_img in enumerate(cocoEval.evalImgs):  # loop through every gt box
            # Only process false positive examples
            for i in range(len(eval_img['dtIds'])):  # det are sorted by scores from high to low
                # matched with gt or det with low confidence
                if eval_img['dtMatches'][0][i] != 0 or eval_img['dtScores'][i] < score_thd:
                    continue
                else:
                    det_id = eval_img['dtIds'][i]
                    det_box = cocoEval.cocoDt.anns[det_id]['bbox']

                    # cat ID of gt box also means the cat ID of det box
                    imgId, catId = eval_img['image_id'], eval_img['category_id']

                    det_info = {
                        'id': count,
                        'image_id': imgId,
                        'bbox': det_box,
                        'area': det_box[2] * det_box[3],
                        'category_id': catId,
                        # "iscrowd": 0,
                    }
                    count += 1
                    result.append(det_info)

                    if i >= max_det:
                        break

        print(f'{count} hard negative detections saved for {len(cocoEval.evalImgs)} images')
        return result
