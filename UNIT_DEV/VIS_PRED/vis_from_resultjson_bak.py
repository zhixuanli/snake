from lib.config import cfg, args
import numpy as np
import pdb
import warnings

from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import torch
from itertools import cycle
import os

mean = snake_config.mean
std = snake_config.std


class Visualizer:
    def visualize_ex(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color)

        plt.show()

    def visualize_training_box(self, output, batch, index):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio

        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=3)

            x_min, y_min, x_max, y_max = box[i]
            ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='w', linewidth=0.5)

        plt.show()
        # os.mkdir()
        os.makedirs("KINS_SNAKE_VIS/", exist_ok=True)
        out_png_path = "KINS_SNAKE_VIS/" + str(index) + ".jpg"

        # plt.savefig(out_png_path, format='jpg', transparent=True, dpi=100, pad_inches = 0)
        plt.savefig(out_png_path, format='jpg', transparent=True, dpi=300, bbox_inches='tight')

    def visualize_gt(self, output, batch, index):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio

        ex = output['i_gt_4py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=3)

            x_min, y_min, x_max, y_max = box[i]
            ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='w', linewidth=0.5)

        plt.show()
        # os.mkdir()
        os.makedirs("KINS_SNAKE_GT_VIS/", exist_ok=True)
        out_png_path = "KINS_SNAKE_GT_VIS/" + str(index) + ".jpg"

        # plt.savefig(out_png_path, format='jpg', transparent=True, dpi=100, pad_inches = 0)
        plt.savefig(out_png_path, format='jpg', transparent=True, dpi=100, bbox_inches='tight')

    def visualize(self, output, batch, index):
        # self.visualize_ex(output, batch)
        self.visualize_training_box(output, batch, index)
        # self.visualize_gt(output, batch, index)


def run_visualize():
    from lib.datasets import make_data_loader
    import tqdm
    import torch

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = Visualizer()
    index = 0

    coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))

    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        # pdb.set_trace()
        visualizer.visualize(output, batch, index)

        index = index + 1


if __name__ == '__main__':
    """python run.py --type visualize --cfg_file configs/kins_snake.yaml test.dataset KinsVal ct_score 0.3"""
    warnings.filterwarnings('ignore')
    globals()['run_'+args.type]()

    args.cfg_file = "configs/kins_snake.yaml"
    run_visualize()
