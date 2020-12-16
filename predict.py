# -*- coding: utf-8 -*-
# @Time    : 2020/12/07 15:04
# @Author  : liyujun

from torchvision import transforms
import time
import torch
import os
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import copy


class ResizeShortSize:
    def __init__(self, config):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        """
        self.short_size = config.resize_short_size

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        """
        im = data['img']
        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 保证短边 >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        data['img'] = im
        return data


class SegDetectorRepresenter():
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch, pred, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        '''
        print(pred.size())
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)          # 语义分割后图
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.size(0)):
            height, width = batch['shape'][batch_index]
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []

        _, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        _, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


class PredictConfig(object):
    def __init__(self):
        super(PredictConfig, self).__init__()
        self.model_path = "./output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best_model.pth"
        self.resize_short_size = 1024
        self.img_mode = "RGB"
        self.use_cuda = True

        self.post_processing_thresh = 0.3
        self.post_processing_box_thresh = 0.7
        self.post_processing_max_candidates = 1000
        self.post_processing_unclip_ratio = 1.5


class PredictModel(object):
    def __init__(self):
        super(PredictModel, self).__init__()
        self.config = PredictConfig()                               # 参数配置类
        self.resize_image = ResizeShortSize(self.config)            # 图片缩放类

        self.transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.model = torch.load(self.config.model_path)             # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.post_processing = SegDetectorRepresenter(
            thresh=self.config.post_processing_thresh,
            box_thresh=self.config.post_processing_box_thresh,
            max_candidates=self.config.post_processing_max_candidates,
            unclip_ratio=self.config.post_processing_unclip_ratio
        )

    def predict(self, img_path, is_output_polygon=False):
        assert os.path.exists(img_path), "file is not exists"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.config.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_copy = copy.deepcopy(img)
        height, weight = img.shape[:2]
        data = {"img": img}
        batch = {'shape': [(height, weight)]}
        data = self.resize_image(data=data)
        img_tensor = self.transformer(data["img"])
        img_tensor = torch.unsqueeze(img_tensor, dim=0)                         # 扩展batch的维度
        img_tensor = img_tensor.to(self.device)                              # 数据拷贝
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(img_tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_processing(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t, img_copy


def draw_box_on_img(img, text_polys):
    img_copy = copy.deepcopy(img)
    text_polys_copy = copy.deepcopy(text_polys)
    for box in text_polys_copy:
        box_reshape = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [box_reshape], True, (0, 0, 255), 2)
    cv2.imshow("image_before", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == "__main__":
    predict_model = PredictModel()
    _, box_list, score_list, t, img = predict_model.predict("./test.jpg")
    print(box_list)
    print(score_list)
    print(t)
    draw_box_on_img(img, text_polys=box_list)