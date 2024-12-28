# create date: 2022/11/11
# author: Dongdong Sun
# email: ddongSun@mail.hfut.edu.cn

import os
import random
import time
import csv
import numpy as np
import torch
import torch.utils.data as data


def left(bag, pos, N):
    max_pos = torch.max(pos, 0, keepdim=True).values
    min_pos = torch.min(pos, 0, keepdim=True).values
    pos = pos - min_pos

    mid_pos = torch.ones((N, 2))
    mid_pos[:, 0] = max_pos[:, 1] - pos[:, 1]
    mid_pos[:, 1] = pos[:, 0]
    pos = mid_pos
    auxi_value = pos[:, 0] * 10000 + pos[:, 1]
    sort_result = torch.sort(auxi_value)
    pos = torch.index_select(pos, 0, sort_result.indices)
    bag = torch.index_select(bag, 0, sort_result.indices)
    div_pos = max_pos - min_pos
    div_pos[:, 0], div_pos[:, 1] = div_pos[:, 1], div_pos[:, 0]
    pos = torch.div(pos, div_pos)
    if div_pos[:, 0] == div_pos[:, 1]:
        return bag, pos
    elif div_pos[:, 0] > div_pos[:, 1]:
        scale = div_pos[:, 1] / div_pos[:, 0]
        pos[:, 1] = pos[:, 1] * scale
        return bag, pos
    elif div_pos[:, 1] > div_pos[:, 0]:
        scale = div_pos[:, 0] / div_pos[:, 1]
        pos[:, 0] = pos[:, 0] * scale
        return bag, pos
    else:
        raise Exception("pos error")


def right(bag, pos, N):
    max_pos = torch.max(pos, 0, keepdim=True).values
    min_pos = torch.min(pos, 0, keepdim=True).values
    pos = pos - min_pos

    mid_pos = torch.ones((N, 2))
    mid_pos[:, 0] = pos[:, 1]
    mid_pos[:, 1] = max_pos[:, 0] - pos[:, 0]
    pos = mid_pos
    auxi_value = pos[:, 0] * 10000 + pos[:, 1]
    sort_result = torch.sort(auxi_value)
    pos = torch.index_select(pos, 0, sort_result.indices)
    bag = torch.index_select(bag, 0, sort_result.indices)
    div_pos = max_pos - min_pos
    div_pos[:, 0], div_pos[:, 1] = div_pos[:, 1], div_pos[:, 0]
    pos = torch.div(pos, div_pos)
    if div_pos[:, 0] == div_pos[:, 1]:
        return bag, pos
    elif div_pos[:, 0] > div_pos[:, 1]:
        scale = div_pos[:, 1] / div_pos[:, 0]
        pos[:, 1] = pos[:, 1] * scale
        return bag, pos
    elif div_pos[:, 1] > div_pos[:, 0]:
        scale = div_pos[:, 0] / div_pos[:, 1]
        pos[:, 0] = pos[:, 0] * scale
        return bag, pos
    else:
        raise Exception("pos error")

def buttom(bag, pos, N):
    max_pos = torch.max(pos, 0, keepdim=True).values
    min_pos = torch.min(pos, 0, keepdim=True).values
    pos = pos - min_pos

    mid_pos = torch.ones((N, 2))
    mid_pos[:, 0] = max_pos[:, 0] - pos[:, 0]
    mid_pos[:, 1] = max_pos[:, 1] - pos[:, 1]
    pos = mid_pos
    auxi_value = pos[:, 0] * 10000 + pos[:, 1]
    sort_result = torch.sort(auxi_value)
    pos = torch.index_select(pos, 0, sort_result.indices)
    bag = torch.index_select(bag, 0, sort_result.indices)
    div_pos = max_pos - min_pos
    pos = torch.div(pos, div_pos)
    if div_pos[:, 0] == div_pos[:, 1]:
        return bag, pos
    elif div_pos[:, 0] > div_pos[:, 1]:
        scale = div_pos[:, 1] / div_pos[:, 0]
        pos[:, 1] = pos[:, 1] * scale
        return bag, pos
    elif div_pos[:, 1] > div_pos[:, 0]:
        scale = div_pos[:, 0] / div_pos[:, 1]
        pos[:, 0] = pos[:, 0] * scale
        return bag, pos
    else:
        raise Exception("pos error")


def unchange(bag, pos, N):
    max_pos = torch.max(pos, 0, keepdim=True).values
    min_pos = torch.min(pos, 0, keepdim=True).values
    pos = pos - min_pos
    div_pos = max_pos - min_pos
    pos = torch.div(pos, div_pos)
    if div_pos[:, 0] == div_pos[:, 1]:
        return bag, pos
    elif div_pos[:, 0] > div_pos[:, 1]:
        scale = div_pos[:, 1] / div_pos[:, 0]
        pos[:, 1] = pos[:, 1] * scale
        return bag, pos
    elif div_pos[:, 1] > div_pos[:, 0]:
        scale = div_pos[:, 0] / div_pos[:, 1]
        pos[:, 0] = pos[:, 0] * scale
        return bag, pos
    else:
        raise Exception("pos error")



class TrainBagLoader(data.Dataset):
    def __init__(self, path, data_csv, n_classes, sample_num=100000):
        super(TrainBagLoader, self).__init__()
        self.path = path
        self.n_classes = n_classes
        self.sample_num = sample_num
        self.data_list = []
        f = open(data_csv, 'r')
        self.cls_list = []
        for i in f.readlines():
            slide_feature_path = i.split(',')[0]
            slide_label = int(i.split(',')[-1])
            self.data_list.append((slide_feature_path, slide_label))
            self.cls_list.append(slide_label)
        f.close()

    def __getitem__(self, index: int):
        bag_data = torch.load(os.path.join(self.path, self.data_list[index][0] + "_features.pth"))
        # pos = torch.load(os.path.join(self.path, self.data_list[index][0] + "_coordinates.pth"))
        # pos = pos.float()
        N, _ = bag_data.shape
        if N > 500:
            select_index = torch.LongTensor(random.sample(range(N), 500))
            bag_data = torch.index_select(bag_data, 0, select_index)
            # pos = torch.index_select(pos, 0, select_index)
        #     N = 500
        # m = random.randint(1, 40)
        # if m <= 10:
        #     bag_data, new_pos = unchange(bag_data, pos, N)
        # elif 10 < m <= 20:
        #     bag_data, new_pos = left(bag_data, pos, N)
        # elif 20 < m <= 30:
        #     bag_data, new_pos = right(bag_data, pos, N)
        # else:
        #     bag_data, new_pos = buttom(bag_data, pos, N)
        # bag_data, new_pos = unchange(bag_data, pos, N)
        label = torch.LongTensor([self.data_list[index][1]])
        # print(label)
        return bag_data, label

    def __len__(self):
        return len(self.data_list)

    def get_weights(self):

        labels = np.array(self.cls_list)
        tmp = np.bincount(labels)
        weights = 1 / np.array(tmp[labels], np.float)
        return weights

class ValBagLoader(data.Dataset):
    def __init__(self, path, data_csv, n_classes, sample_num=100000):
        super(ValBagLoader, self).__init__()
        self.path = path
        self.n_classes = n_classes
        self.sample_num = sample_num
        self.data_list = []
        f = open(data_csv, 'r')
        for i in f.readlines():
            slide_feature_path = i.split(',')[0]
            slide_label = int(i.split(',')[-1])
            self.data_list.append((slide_feature_path, slide_label))
        f.close()

    def __getitem__(self, index: int):
        bag_data = torch.load(os.path.join(self.path, self.data_list[index][0] + "_features.pth"))
        # pos = torch.load(os.path.join(self.path, self.data_list[index][0] + "_coordinates.pth"))
        # pos = pos.float()
        N = bag_data.shape[0]
        # bag_data, new_pos = unchange(bag_data, pos, N)
        label = torch.LongTensor([self.data_list[index][1]])

        return bag_data, label

    def __len__(self):
        return len(self.data_list)


class TestBagLoader(data.Dataset):
    def __init__(self, path, data_csv, n_classes, sample_num=100000):
        super(TestBagLoader, self).__init__()
        self.path = path
        self.n_classes = n_classes
        self.sample_num = sample_num
        self.data_list = []
        f = open(data_csv, 'r')
        for i in f.readlines():
            slide_feature_path = i.split(',')[0]
            slide_label = int(i.split(',')[-1].strip())
            self.data_list.append((slide_feature_path, slide_label))
        # print(self.data_list)
        f.close()

    def __getitem__(self, index: int):
        bag_data = torch.load(os.path.join(self.path, self.data_list[index][0] + "_features.pth"))
        # bag_data = torch.load(os.path.join(self.path, self.data_list[index][0] + ".pth"))
        # pos = torch.load(os.path.join(self.path, self.data_list[index][0] + "_coordinates.pth"))
        N, _ = bag_data.shape
        # pos = pos.float()
        # bag_data, new_pos = unchange(bag_data, pos, N)
        label = torch.LongTensor([self.data_list[index][1]])
        # return bag_data, label, new_pos

        return bag_data, label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    for i in range(30):
        print(random.randint(1, 30))
