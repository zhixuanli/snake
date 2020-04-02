import os

from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

json_file = 'instances_val.json'
dataset_dir = '/Users/lizhixuan/PycharmProjects/amodal_dataset/KINS/image/testing/'

coco = COCO(json_file)
catIds = coco.getCatIds(catNms=['person']) # catIds=1 表示人这一类
imgIds = coco.getImgIds(catIds=catIds ) # 图片id，许多值

for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    I = io.imread(dataset_dir + img['file_name'])
    plt.figure(figsize=(10, 5))

    plt.axis('off')
    plt.imshow(I)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    # plt.show()
    save_file_name = "KINS_GT_VIS/" + img['file_name'][:-4]+"_vis.png"
    plt.savefig(save_file_name, transparent=True, dpi=300, pad_inches = 0)
    exit()
