import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def draw_box_on_img(img, text_polys):
    img_copy = copy.deepcopy(img)
    text_polys_copy = copy.deepcopy(text_polys)
    for box in text_polys_copy:
        box_reshape = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [box_reshape], True, (0, 0, 255), 2)
    cv2.imshow("image_before", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


class ImageAug(object):
    def __call__(self, data):
        img = data["img"]                   # 取出图片数据
        text_polys = data["text_polys"]     # 取出文本框对应的点

        # 绘制box在原图中
        # draw_box_on_img(img, text_polys)

        # imgaug进行数据增强
        # 定义变换序列
        seq = iaa.Sequential([iaa.SomeOf((2, 3), [iaa.Flipud(0.5),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Resize(size=(0.5, 3))])
        ])
        seq_det = seq.to_deterministic()    # 固定变换序列

        img_aug = seq_det.augment_image(img)
        text_polys_seq = []
        for box_points in text_polys:
            # 依次变换各个Box
            keypoints = ia.KeypointsOnImage([ia.Keypoint(point[0], point[1]) for point in box_points], shape=img.shape)
            keypoints = seq_det.augment_keypoints([keypoints])[0].keypoints
            poly = np.array([(p.x, p.y) for p in keypoints])
            text_polys_seq.append(poly)

        # draw_box_on_img(img_aug, text_polys_seq)

        data["img"] = img_aug
        data["text_polys"] = np.array(text_polys_seq)
        # print(data["text_polys"].shape)

        return data







