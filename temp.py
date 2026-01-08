import numpy as np
import torch


def inspect_data(name, data):
    for idx, item in np.ndenumerate(data):
        if isinstance(item, dict):
            print("idx:", idx, "item 是 dict/OrderedDict，长度:", len(item))
            for k, v in item.items():
                print("   key:", k, "=>", "value:", v, "类型:", type(v))
        else:
            print("idx:", idx, "item:", item, "类型:", type(item))
    return


def main():
    path1 = "/data1/yueyi/data/amass/KIT-smpl/183/turn_left01_poses.npy"
    path2 = "/data1/yueyi/data/amass/KIT/183/turn_left01_poses.npz"

    data1 = np.load(path1, allow_pickle=True)
    data2 = np.load(path2, allow_pickle=True)

    for idx, item in np.ndenumerate(data1):
        if isinstance(item, dict):
            print("idx:", idx, "item 是 dict/OrderedDict，长度:", len(item))
            for k, v in item.items():
                print("   key:", k, "=>", "value:", v, "类型:", type(v))
        else:
            print("idx:", idx, "item:", item, "类型:", type(item))
    return


if __name__ == "__main__":
    main()
