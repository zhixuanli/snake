"""
    load the 1 class(object class) COCO 14 amodal dataset into amodal masks and their bboxes

    this dataset was released by Yan et. al, Semantic Amodal Segmentation, CVPR 2017
"""
import os
import numpy as np
import json
import cv2


def load_kins_dict(dict_path):
    json_file = dict_path

    with open(json_file) as f:
        imgs_anns = json.load(f)

    pass
    print("int")




if __name__ == '__main__':
    dict_path = "annotations/instances_train.json"

    load_kins_dict(dict_path)
