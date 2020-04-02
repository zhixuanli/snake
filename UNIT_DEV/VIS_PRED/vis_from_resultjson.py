import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils

parent_path = '/Users/lizhixuan/PycharmProjects/amodal_dataset/KINS/image/testing'
json_file = 'results.json'
with open(json_file) as anno_:
    annotations = json.load(anno_)

def apply_mask(image, segmentation):
    alpha = 0.5
    color = (0, 0.6, 0.6)
    threshold = 0.5
    mask = maskUtils.decode(segmentation) # 分割解码
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)
    for c in range(3): # 3个通道
        # mask=1执行前一个，否则后一个
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

results = []
for i in range(len(annotations)):
    annotation = annotations[i]
    image_id = annotation['image_id']
    # 包含size:图片高度宽度  counts:压缩后的mask  通过mask = maskUtils.decode(encoded_mask)解码，得到mask,需要导入from pycocotools import mask as maskUtils
    segmentation = annotation['segmentation']
    full_path = os.path.join(parent_path, str(image_id).zfill(6) + '.png')
    image = cv2.imread(full_path)
    mask_image = apply_mask(image, segmentation)
    cv2.imshow('demo', mask_image)
    cv2.waitKey(5000)
