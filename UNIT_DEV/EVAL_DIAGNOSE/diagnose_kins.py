from lib.config import cfg, args
import os
import cv2
import json
import numpy as np
from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils
from external.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
import pycocotools.mask as mask_util
import pycocotools.coco as coco
from pycocotools.cocoeval_diagnose import COCOeval
# from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


class Evaluator:
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.data_root = args['data_root']
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio

        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)

        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)

        coco_dets = []
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def delete_all_small_area_gt(self, coco_eval):
        small_area_thresh = 100

        return coco_eval

    def _count_statistic_for_diff_size(self, area_all, img_area_origin, thresh1_p, thresh2_p):
        thresh_1 = img_area_origin * thresh1_p
        thresh_2 = img_area_origin * thresh2_p
        area_local = area_all.copy()
        area_local[area_local < thresh_1] = 0
        area_local[area_local > thresh_2] = 0
        area_local[area_local != 0] = 1
        count_ = sum(area_local)

        return count_

    def statistic_area_in_gt_briefly(self, coco_eval):
        instance_count = len(coco_eval.cocoGt.anns)
        area_all = []
        for i in range(instance_count):
            anns_ = coco_eval.cocoGt.anns[i]
            area_ = anns_["area"]
            area_all.append(area_)

        area_all.sort()
        area_all = np.asarray(area_all)

        img_h = coco_eval.cocoGt.imgs[0]["height"]
        img_w = coco_eval.cocoGt.imgs[0]["width"]
        img_area_origin = img_h * img_w

        size_statistic_count = []
        # extra small 0-10%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0, 0.1)
        size_statistic_count.append(count_)

        # small 10-30%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0.1, 0.3)
        size_statistic_count.append(count_)

        # medium 30-70%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0.3, 0.7)
        size_statistic_count.append(count_)

        # large 70-90%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0.7, 0.9)
        size_statistic_count.append(count_)

        # extra-large 90-100%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0.9, 1)
        size_statistic_count.append(count_)

        print("extra_small  small   medium  large   extra_large")
        print("%d\t%d\t%d\t%d\t%d" % (size_statistic_count[0], size_statistic_count[1], size_statistic_count[2],
                                      size_statistic_count[3], size_statistic_count[4]))
        print("%f\t%f\t%f\t%f\t%f" % (size_statistic_count[0]/instance_count,
                                      size_statistic_count[1]/instance_count,
                                      size_statistic_count[2]/instance_count,
                                      size_statistic_count[3]/instance_count,
                                      size_statistic_count[4]/instance_count))

        print("max area: "+str(max(area_all)))
        print("min area: "+str(min(area_all)))
        print("totoal instance count: " + str(instance_count))
        # plt.hist(area_all)
        # plt.hist(area_all, bins=100, alpha=0.5, histtype='stepfilled',
        #          color='steelblue', edgecolor='none')
        # plt.show()

        # pass

    def statistic_area_in_gt_detailed_for_extra_small(self, coco_eval):
        instance_count = len(coco_eval.cocoGt.anns)
        area_all = []
        for i in range(instance_count):
            anns_ = coco_eval.cocoGt.anns[i]
            area_ = anns_["area"]
            area_all.append(area_)

        area_all.sort()
        area_all = np.asarray(area_all)

        img_h = coco_eval.cocoGt.imgs[0]["height"]
        img_w = coco_eval.cocoGt.imgs[0]["width"]
        img_area_origin = img_h * img_w

        size_statistic_count = []
        # extra small 0-10%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0, 0.01)
        size_statistic_count.append(count_)

        # small 10-30%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0.01, 0.03)
        size_statistic_count.append(count_)

        # medium 30-70%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0.03, 0.07)
        size_statistic_count.append(count_)

        # large 70-90%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0.07, 0.09)
        size_statistic_count.append(count_)

        # extra-large 90-100%
        count_ = self._count_statistic_for_diff_size(area_all, img_area_origin, 0.09, 0.1)
        size_statistic_count.append(count_)

        area_all[area_all > 12000] = 0
        area_tmp = area_all.copy()
        area_tmp[area_tmp != 0] = 1
        area_focus = area_all[0:int(sum(area_tmp)-1)]

        print("extra_small\tsmall\tmedium\tlarge\textra_large")
        print("%d\t%d\t%d\t%d\t%d" % (size_statistic_count[0], size_statistic_count[1], size_statistic_count[2],
                                      size_statistic_count[3], size_statistic_count[4]))
        print("%f\t%f\t%f\t%f\t%f" % (size_statistic_count[0]/instance_count,
                                      size_statistic_count[1]/instance_count,
                                      size_statistic_count[2]/instance_count,
                                      size_statistic_count[3]/instance_count,
                                      size_statistic_count[4]/instance_count))

        print("max area: "+str(max(area_focus)))
        print("min area: "+str(min(area_focus)))
        print("origin img area: " + str(img_area_origin))
        print("totoal instance count: " + str(instance_count))

        plt.hist(area_focus, bins=60, alpha=0.5,
                 color='steelblue', edgecolor='none')
        plt.savefig("detailed_statistic_in_extreme_small.png", bbox_inches="tight")
        plt.show()

    def summarize(self):
        # json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'segm')

        # statistic the distribution in different area
        # self.statistic_area_in_gt_briefly(coco_eval)

        # statistic the distribution in extra small area
        # self.statistic_area_in_gt_detailed_for_extra_small(coco_eval)
        # exit()

        img_count = len(coco_eval.cocoGt.imgs)
        # img_count = 50
        imgid_with_metric = np.zeros([img_count, 2])

        self.img_ids = []
        # for i in range(img_count):
        for i in range(img_count):
            self.img_ids.append(i)
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        for i in range(img_count):
            print("[ %d / %d ] evaluating...." % (i + 1, img_count))
            coco_eval.accumulate_for_single_img(imgid=i)
            stats = coco_eval.summarize_single_img()
            ap_single = stats[0]
            imgid_with_metric[i, 0] = i
            imgid_with_metric[i, 1] = ap_single
        print(imgid_with_metric)
        np.save("imgid_with_metric.npy", imgid_with_metric)

        # coco_eval.accumulate()
        # coco_eval.summarize()
        # self.results = []
        # self.img_ids = []
        # self.aps.append(coco_eval.stats[0])
        # return {'ap': coco_eval.stats[0]}


def run_evaluate():
    evaluator = Evaluator(cfg.result_dir)
    evaluator.summarize()




if __name__ == '__main__':
    "python run.py --type evaluate test.dataset $dataset_name resume True model $model_name task $task"

    args.cfg_file = "configs/kins_snake.yaml"
    run_evaluate()
    # args.type = "test.dataset"

    # test.dataset
    # KinsVal

