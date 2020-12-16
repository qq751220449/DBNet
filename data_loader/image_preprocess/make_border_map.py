import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pyclipper
from shapely.geometry import Polygon
from data_loader.image_preprocess.iaa_augment import draw_box_on_img           # 导入图片绘制工具


class MakeBorderMap(object):
    def __init__(self, config):
        self.shrink_ratio = config.make_border_shrink_ratio     # 收缩因子,经验设置为0.4
        self.thresh_min = config.make_border_thresh_min         # 生成的概率图的最小值
        self.thresh_max = config.make_border_thresh_max         # 生成的概率图的最大值

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']
        # print(ignore_tags)
        # draw_box_on_img(im, text_polys)

        canvas = np.zeros(im.shape[:2], dtype=np.float32)
        mask = np.zeros(im.shape[:2], dtype=np.float32)

        for i in range(len(text_polys)):
            if ignore_tags[i]:
                # 忽略不清楚的文本
                continue
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        # cv2.imshow("threshold_map", canvas)
        # cv2.imshow("threshold_mask", mask)
        # cv2.waitKey(0)
        data['threshold_map'] = canvas
        data['threshold_mask'] = mask

        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2        # 判断维度
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)        # 基于当前点构建多边形
        if polygon_shape.area <= 0:
            return
        # 根据面积与周长计算收缩偏移量  D=A(1-r*r)/L   见DB论文
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        # https://github.com/fonttools/pyclipper 参考连接

        padded_polygon = np.array(padding.Execute(distance)[0])         # 向外扩展
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)      # padded_polygon收缩之后的点坐标

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin        # 将原始文本框移动到图片的左上角
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))        # 生成坐标系
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]      # 依次处理线段
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
            # cv2.imshow("test", distance_map[i])
            # cv2.waitKey(0)
        distance_map = distance_map.min(axis=0)
        # cv2.imshow("distance_map", distance_map)
        # cv2.waitKey(0)
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])       # 移动处理好的文本框到图像原本的位置

        # cv2.imshow("canvas", canvas)
        # cv2.waitKey(0)

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))        # 三角余弦公式  a^2=b^2+c^2-2bc*cosA  边a的对角为A
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)

        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)      # 计算三角形的面积之后再计算 距离   b*sinA就是高,乘以c就是面积
        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]       # 注意三角余弦公式中的负号
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2
