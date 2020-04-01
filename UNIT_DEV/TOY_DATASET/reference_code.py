"""
    load the 1 class(object class) COCO 14 amodal dataset into amodal masks and their bboxes

    this dataset was released by Yan et. al, Semantic Amodal Segmentation, CVPR 2017
"""
import os
import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import itertools
import random
from detectron2.utils.visualizer import Visualizer
import cv2


# write a function that loads the dataset into detectron2's standard format
def get_COCO14_modal_dicts(root_dir, mode="train", year="2014"):
    json_dir = root_dir + "annotations/"
    img_dir = root_dir + mode + year
    json_file = json_dir + "COCOA_amodal_{}{}.json".format(mode, year)
    # json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    images_list = imgs_anns["images"]
    annotations_list = imgs_anns["annotations"]

    dataset_dicts = []
    for i in range(len(images_list)):
        img_info = images_list[i]
        anno_info = annotations_list[i]

        record = {}
        # record image information
        filename = os.path.join(img_dir, img_info["file_name"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = img_info["id"]
        record["height"] = height
        record["width"] = width

        annos = anno_info["regions"]
        # record annotations information
        objs = []
        for anno in annos:
            seg_poly = anno["segmentation"]
            px = seg_poly[::2]
            py = seg_poly[1::2]

            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


if __name__ == '__main__':
    for d in ["train", "val"]:
        DatasetCatalog.register("COCO_amodal_" + d,
                                lambda d=d: get_COCO14_modal_dicts("../../datasets/cocoa/",
                                                                   mode=d, year="2014"))
        MetadataCatalog.get("COCO_amodal_" + d).set(thing_classes=["objects"])
    balloon_metadata = MetadataCatalog.get("COCO_amodal")

    dataset_dicts = get_COCO14_modal_dicts("../../datasets/cocoa/", mode="train", year="2014")

    for i in range(len(dataset_dicts)):
        print("[ {} / {} ] saving amodal gt vis".format(i, len(dataset_dicts)))
        d = dataset_dicts[i]
        img_id = d["image_id"]
        # if img_id != 6773:
        #     continue
        img = cv2.imread(d["file_name"])

        img_name = d["file_name"].split("/")[-1]

        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=1.5)
        vis = visualizer.draw_dataset_dict(d)

        vis_img = vis.get_image()[:, :, ::-1]
        # cv2.imshow("amodal gt vis", vis_img)
        cv2.imwrite("amodal_gt_vis/train/amodal_gt_vis"+img_name, vis_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # exit()
