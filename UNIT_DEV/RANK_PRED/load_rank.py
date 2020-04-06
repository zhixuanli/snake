import numpy as np


if __name__ == '__main__':
    rank_ = np.load("imgid_with_metric.npy")
    with open("rank.txt", "a") as f:
        for i in range(rank_.shape[0]):
            lines_to_write = str(rank_[i][0]) + "\t" + str(rank_[i][1]) + "\n"
            f.writelines(lines_to_write)
    print(rank_.shape)
