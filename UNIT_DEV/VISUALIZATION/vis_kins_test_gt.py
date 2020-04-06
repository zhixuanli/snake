"""
    save COCOA val gt
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
import pickle


# write a function that loads the dataset into detectron2's standard format
def get_COCO14_modal_dicts(json_file_path, img_dir):

    dict_path = "kins_gt_dict.pkl"
    dict_name = "kins_gt_dict"
    if os.path.exists(dict_path):
        print("DICT IS PRECOMPUTED, LOADING NOW")
        dataset_dicts = load_obj(dict_name)

        return dataset_dicts

    print("Starting to get kins ")
    with open(json_file_path) as f:
        imgs_anns = json.load(f)

    images_list = imgs_anns["images"]
    annotations_list = imgs_anns["annotations"]

    dataset_dicts = []
    anno_list_idx_current = 0
    for i in range(len(images_list)):
        print("processing [ %d / %d ]..." % (i, len(images_list)))
        img_info = images_list[i]

        record = {}
        # record image information
        filename = os.path.join(img_dir, img_info["file_name"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = img_info["id"]
        record["height"] = height
        record["width"] = width

        # record annotations information
        objs = []
        current_img_id = img_info["id"]

        anno_info = annotations_list[anno_list_idx_current]
        while(anno_info["image_id"]==current_img_id):
            seg_poly = anno_info["segmentation"][0]
            px = seg_poly[::2]
            py = seg_poly[1::2]

            # poly = [(x, y) for x, y in zip(px, py)]
            # poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": anno_info["segmentation"],
                "category_id": anno_info["category_id"]-1,
            }
            objs.append(obj)
            anno_list_idx_current += 1
            if anno_list_idx_current == len(annotations_list):
                break
            anno_info = annotations_list[anno_list_idx_current]

        record["annotations"] = objs
        dataset_dicts.append(record)

        # anno_list_idx_current += 1

    save_obj(dataset_dicts, dict_name)

    return dataset_dicts


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def register_dataset():
    for d in ["val"]:
        DatasetCatalog.register("KINS_" + d,
                                lambda d=d: get_COCO14_modal_dicts("instances_val.json",
                                                                   "/Users/lizhixuan/PycharmProjects/amodal_dataset/KINS/image/testing"))
        MetadataCatalog.get("KINS_" + d).set(thing_classes=['cyclist','pedestrian',"no", 'car', "tram", "truck", "van", "misc"])
    print("REGISTERED!")


if __name__ == '__main__':
    for d in ["val"]:
        DatasetCatalog.register("KINS_" + d,
                                lambda d=d: get_COCO14_modal_dicts("instances_val.json",
                                                                   "/Users/lizhixuan/PycharmProjects/amodal_dataset/KINS/image/testing"))
        MetadataCatalog.get("KINS_" + d).set(thing_classes=['cyclist','pedestrian',"no", 'car', "tram", "truck", "van", "misc"])
    balloon_metadata = MetadataCatalog.get("KINS_val")

    dataset_dicts = get_COCO14_modal_dicts("instances_val.json",
                                           "/Users/lizhixuan/PycharmProjects/amodal_dataset/KINS/image/testing")

    for i in range(len(dataset_dicts)):
        print("[ {} / {} ] saving KINS gt vis".format(i, len(dataset_dicts)))
        d = dataset_dicts[i]
        img_id = d["image_id"]
        img = cv2.imread(d["file_name"])

        img_name = d["file_name"].split("/")[-1]

        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=1, edge_width=2, ifDrawBox=False)
        vis = visualizer.draw_dataset_dict(d, alpha=0)

        vis_img = vis.get_image()[:, :, ::-1]
        cv2.imwrite("VIS_KINS_TEST_GT/vis_"+img_name, vis_img)

        # cv2.imshow("amodal gt vis", vis_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # if i == 5:
        #     exit()
