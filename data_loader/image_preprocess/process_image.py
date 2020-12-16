import os
import cv2
import copy
import numpy as np
from utils.util import order_points_clockwise
from torch.utils.data import Dataset


class ICDAR2015Dataset(Dataset):
    # ICDAR2015数据集
    def __init__(self, params, mode="train", transform=None, img_transform=None):
        super(ICDAR2015Dataset, self).__init__()

        assert params["img_mode"] in ["RGB", "BGR"]      # 判断图片格式是否满足要求
        assert mode == "train" or mode == "test"
        self.img_mode = params["img_mode"]
        self.ignore_tags = params["ignore_tags"]

        if mode == "train":
            self.data_list = self.load_data(params["train_dataset_path"])
            self.filter_keys = params["train_filter_keys"]
        elif mode == "test":
            self.data_list = self.load_data(params["test_dataset_path"])
            self.filter_keys = params["test_filter_keys"]

        self.transform = transform      # 变换方式
        self.img_transform = img_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # 返回索引样本
        data = copy.deepcopy(self.data_list[item])
        img = cv2.imread(data["img_path"], cv2.IMREAD_COLOR)
        if self.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data["img"] = img
        data['shape'] = [img.shape[0], img.shape[1]]
        if self.img_transform is not None:      # 数据增强及预处理
            data = self.img_transform(data)

        if self.transform is not None:
            data["img"] = self.transform(data["img"])

        data['text_polys'] = data['text_polys'].tolist()

        if len(self.filter_keys):
            data_dict = {}
            for key, value in data.items():
                if key not in self.filter_keys:
                    data_dict[key] = value
            return data_dict
        else:
            return data

    def load_data(self, dataset_path):
        """
        加载训练文件中文件目录
        :param dataset_path:待加载的数据集目录
        :return:
        """
        # 打开训练数据集文件
        data_list = []
        with open(dataset_path, "r", encoding="utf-8") as fid:
            for line in fid.readlines():
                image_path, label_path = line.strip().split("\t")
                data_list.append((image_path, label_path))
        # print(data_list)
        t_data_list = []
        for image_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data["text_polys"]) > 0:
                item = {
                    "img_path": image_path,
                    "img_name": os.path.split(image_path)[1]
                        }
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path):
        # print(label_path)
        boxes = []      # 存放box列表
        texts = []      # 文本数据列表
        ignores = []    # 文本忽略标志
        with open(label_path, "r", encoding="utf-8") as fid:
            for line in fid.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = ",".join(params[8:])
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)       # 标志文本内容不清晰的文本框
                except:
                    print("load label failed on {}".format(label_path))
        data = {
            "text_polys": np.array(boxes),       # box列表
            "text": texts,
            "ignore_tags": ignores
        }
        return data


