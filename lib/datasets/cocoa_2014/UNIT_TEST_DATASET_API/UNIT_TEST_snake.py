import os
# from lib.utils.snake import snake_coco_utils, snake_config, visualize_utils
import cv2
import numpy as np
# import math
# from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.anns = sorted(self.coco.getImgIds())
        self.anns = np.array([ann for ann in self.anns if len(self.coco.getAnnIds(imgIds=ann, iscrowd=None))])
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=0)
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def __getitem__(self, index):
        ann = self.anns[index]

        anno, path, img_id = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, path)

        height, width = img.shape[0], img.shape[1]

    def __len__(self):
        return len(self.anns)


if __name__ == '__main__':
    # UNIT TEST FOR DATASET
    from dataset_catalog import DatasetCatalog

    dataset_name = "cocoa_train"
    args = DatasetCatalog.get(dataset_name)
    del args["id"]

    dataset = Dataset(args["ann_file"], args["data_root"], args["split"])
    print("H")
