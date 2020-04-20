"""
transform COCOA 2014 origin annotation file to standard COCO format

the depth-constraint will lost after transform
"""

import json
import numpy as np


def change_to_instance_index(origin_list):
    new_anno_list = []

    unique_anno_id = 0

    for i in range(len(origin_list)):
        print("[ %d / %d ] processing img..." % (i+1, len(origin_list)))
        img_anno = origin_list[i]
        instance_count = img_anno["size"]
        for k in range(instance_count):
            instance_anno_ = dict()

            anno_local = img_anno["regions"][k]

            instance_anno_["category_id"] = 1  # only 1
            instance_anno_["id"] = unique_anno_id  # I don't know whether this is correct
            unique_anno_id += 1
            instance_anno_["area"] = anno_local["area"]
            instance_anno_["image_id"] = img_anno["image_id"]

            seg_poly = anno_local["segmentation"]
            px = seg_poly[::2]
            py = seg_poly[1::2]

            # wrong: "bbox_mode" here: BoxMode.XYXY_ABS
            # right: "bbox_mode" here: XYWH
            instance_anno_["bbox"] = [np.min(px), np.min(py), np.max(px) - np.min(px), np.max(py) - np.min(py)]

            instance_anno_["iscrowd"] = 0  # all is poly, so this should be 0
            instance_anno_["segmentation"] = [anno_local["segmentation"]]

            new_anno_list.append(instance_anno_)

    return new_anno_list


def trans_to_COCO_format(annotation_file_path, coco_format_dict_save_name):
    dataset = json.load(open(annotation_file_path, 'r'))

    # add category info for cocoa
    cat_ = dict()
    cat_["id"] = 1
    cat_["name"] = "object"
    cat_["supercategory"] = "object"

    # add info
    info_ = dict()
    info_["description"] = "coco amodal 2014 dataset, completely conform to COCO STD format"
    info_["year"] = 2014

    dataset["categories"] = [cat_]
    dataset["info"] = info_

    # change annotations from imgid index to instance index
    annotations = dataset["annotations"]
    instance_anno = change_to_instance_index(annotations)
    dataset["annotations"] = instance_anno

    dataset_new = json.dumps(dataset)

    with open(coco_format_dict_save_name, "w") as f:
        f.write(dataset_new)
    print("transform done!")


if __name__ == '__main__':
    dataset_root = "/Users/lizhixuan/PycharmProjects/amodal_dataset/"

    # STD example
    # coco_ = dataset_root + "COCO/annotations/instances_train2014.json"
    # coco_dataset = json.load(open(coco_, "r"))

    # train
    annotation_file = dataset_root + "COCOA/annotations/COCO_amodal_train2014.json"
    coco_new_path = "COCOA_train.json"
    trans_to_COCO_format(annotation_file, coco_new_path)

    # val
    annotation_file = dataset_root + "COCOA/annotations/COCO_amodal_val2014.json"
    coco_new_path = "COCOA_val.json"
    trans_to_COCO_format(annotation_file, coco_new_path)





