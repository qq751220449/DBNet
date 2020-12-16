import cv2


class ResizeShortSize:
    def __init__(self, config):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = config.resize_short_size
        self.resize_text_polys = config.resize_text_polys

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 保证短边 >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            # im, scale = resize_image(im, self.short_size)
            if self.resize_text_polys:
                # text_polys *= scale
                text_polys[:, 0] *= scale[0]
                text_polys[:, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        return data