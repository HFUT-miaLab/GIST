import os
import random
import numpy as np
import torch
import torch.utils.data as data



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

        N, _ = bag_data.shape
        select_index = torch.LongTensor(random.sample(range(N), 500))
        bag_data = torch.index_select(bag_data, 0, select_index)

        label = torch.LongTensor([self.data_list[index][1]])
        return bag_data, label

    def __len__(self):
        return len(self.data_list)

    def get_weights(self):

        labels = np.array(self.cls_list)
        tmp = np.bincount(labels)
        weights = 1 / np.array(tmp[labels], np.float)
        return weights

