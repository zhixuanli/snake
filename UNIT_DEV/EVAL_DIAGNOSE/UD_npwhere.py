import numpy as np

iouThr = 0.1

iou_start = 0
iou_end = 0.95
iou_step = 0.05
iouThrs = np.linspace(iou_start, iou_end, int(np.round((iou_end - iou_start) / iou_step)) + 1, endpoint=True)

print(iouThrs)
print(iouThr == iouThrs)
print(iouThrs[2] == iouThrs)
print(iouThrs[2])

iouThrs = np.around(iouThrs, decimals=2)
print(iouThrs[2])
print(iouThrs)
print(iouThr == iouThrs)

t = np.where(iouThr == iouThrs)[0]
