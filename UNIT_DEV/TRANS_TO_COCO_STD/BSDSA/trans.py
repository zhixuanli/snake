"""
transform BSDSA origin annotation file to standard COCO format

the depth-constraint will lost after transform
"""

import json
import numpy as np

np.random.seed(2020)

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
            instance_anno_["bbox"] = [np.min(px), np.min(py), np.max(px)-np.min(px), np.max(py)-np.min(py)]

            instance_anno_["iscrowd"] = 0  # all is poly, so this should be 0
            instance_anno_["segmentation"] = [anno_local["segmentation"]]

            new_anno_list.append(instance_anno_)

    return new_anno_list


def get_all_category_info_as_dict_list(dataset):
    anno = dataset["annotations"]
    category_list = []
    catergory_set = set()
    for i in range(len(anno)):
        anno_ = anno[i]["regions"]
        for k in range(len(anno_)):
            seg_ = anno_[k]
            catergory_set.add(seg_["name"])

    # there are totally 866 categories in this annotation, which is not reasonable
    # so there is no need to assign category labels


def find_and_random_choose_one_anno(dataset, current_img_id, current_anno_finding_index):
    annos_all = dataset["annotations"]
    anno_for_img_id_all = []

    # find all anno for current img
    for i in range(current_anno_finding_index, len(annos_all)):
        current_anno_img_id = annos_all[i]["image_id"]
        if current_img_id == current_anno_img_id:
            anno_for_img_id_all.append(annos_all[i])
        else:
            current_anno_finding_index = i+1
            break

    # random choose one from them
    len_ = len(anno_for_img_id_all)
    choose_id = np.random.randint(0, len_-1)

    reserve_anno_for_this_img = anno_for_img_id_all[choose_id]

    return reserve_anno_for_this_img, current_anno_finding_index


def reserve_random_choose_anno_for_each_img(dataset):
    """ each img should only have one annotation dict(including many instances)
        we will choose it randomly from multi-annotators
    """
    img_info_list = dataset["images"]
    anno_only_for_single_img = []
    current_finding_index = 0
    for i in range(len(img_info_list)):
        current_img_id = img_info_list[i]["id"]
        reserve_anno_for_this_img, current_finding_index = \
            find_and_random_choose_one_anno(dataset, current_img_id, current_finding_index)
        anno_only_for_single_img.append(reserve_anno_for_this_img)

    dataset["annotations"] = anno_only_for_single_img

    return dataset


def trans_to_COCO_format(annotation_file_path, coco_format_dict_save_name):
    dataset = json.load(open(annotation_file_path, 'r'))

    # add category info for cocoa
    cat_ = dict()
    cat_["id"] = 1
    cat_["name"] = "object"
    cat_["supercategory"] = "object"

    # add info
    info_ = dict()
    info_["description"] = "BSDS amodal 2014 dataset, completely conform to COCO STD format"
    # info_["year"] = 2014
    info_["url"] = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html"

    dataset["categories"] = [cat_]
    dataset["info"] = info_

    dataset = reserve_random_choose_anno_for_each_img(dataset)

    # change annotations from imgid index to instance index
    annotations = dataset["annotations"]
    instance_anno = change_to_instance_index(annotations)
    dataset["annotations"] = instance_anno

    dataset_new = json.dumps(dataset)

    with open(coco_format_dict_save_name, "w") as f:
        f.write(dataset_new)
    print("transform done!")

    return dataset


def merge_train_and_val(train_dataset, val_dataset):
    trainval_dataset = dict()
    trainval_dataset["info"] = train_dataset["info"]
    trainval_dataset["categories"] = train_dataset["categories"]

    # merge images
    trainval_dataset["images"] = train_dataset["images"] + val_dataset["images"]

    # merge annotations, id should be recalculated
    trainval_dataset["annotations"] = train_dataset["annotations"] + val_dataset["annotations"]
    for i in range(len(trainval_dataset["annotations"])):
        trainval_dataset["annotations"][i]["id"] = i

    dataset_new = json.dumps(trainval_dataset)

    with open("BSDSA_trainval.json", "w") as f:
        f.write(dataset_new)
    print("merge done!")


if __name__ == '__main__':
    dataset_root = "/Users/lizhixuan/PycharmProjects/amodal_dataset/"

    # STD example
    # coco_ = dataset_root + "COCO/annotations/instances_train2014.json"
    # coco_dataset = json.load(open(coco_, "r"))

    # train
    annotation_file = dataset_root + "BSDSA/annotations/BSDS_amodal_train.json"
    coco_new_path = "BSDSA_train.json"
    train_dataset = trans_to_COCO_format(annotation_file, coco_new_path)

    # val
    annotation_file = dataset_root + "BSDSA/annotations/BSDS_amodal_val.json"
    coco_new_path = "BSDSA_val.json"
    val_dataset = trans_to_COCO_format(annotation_file, coco_new_path)

    merge_train_and_val(train_dataset, val_dataset)

    # test
    annotation_file = dataset_root + "BSDSA/annotations/BSDS_amodal_test.json"
    coco_new_path = "BSDSA_test.json"
    trans_to_COCO_format(annotation_file, coco_new_path)

    print("ALL IS OK")





