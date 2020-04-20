import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from UNIT_DEV.VISUALIZATION.vis_COCOA_test_gt import register_dataset


def dataset_id_map(id):
    # labels = ['cyclist','pedestrian',"no", 'car', "tram", "truck", "van", "misc"]
    # return labels[id]
    return id-1

def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]

    # if predictions[0]["bbox"] is not None:
    #     bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
    #     bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    # here it should -1 because in kins, category start from 1
    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    # ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    register_dataset()

    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=False, help="JSON file produced by the model", default="results.json")
    parser.add_argument("--output", required=False, help="output directory", default="VIS_COCOA_TEST_PRED")
    parser.add_argument("--dataset", help="name of the dataset", default="COCOA_val")
    parser.add_argument("--conf-threshold", default=0.3, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "rb") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata, edge_width=1, ifShowLabel=False)
        vis_pred = vis.draw_instance_predictions(predictions, alpha=0).get_image()

        vis = Visualizer(img, metadata, edge_width=1, ifDrawBox=False, ifShowLabel=False)
        vis_gt = vis.draw_dataset_dict(dic, alpha=0).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=0)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
