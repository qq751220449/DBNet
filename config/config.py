import os
import argparse
import sys
# root目录设置
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
sys.path.append(base_dir)       # 设置项目根目录


class EvalConfig(object):
    def __init__(self):
        super(EvalConfig, self).__init__()
        """
        校验时参数设置
        """
        self.model_path = "output/DBNet_resnet18_FPN_DBHead/checkpoint/model_latest.pth"
        self.model_path_model = "output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best_model.pth"
        self.post_processing_thresh = 0.3
        self.post_processing_box_thresh = 0.7
        self.post_processing_max_candidates = 1000
        self.post_processing_unclip_ratio = 1.5
        self.post_processing_type = "SegDetectorRepresenter"

        self.metric_type = "QuadMetric"
        self.metric_is_output_polygon = True



class TrainConfig(object):
    def __init__(self):
        super(TrainConfig, self).__init__()
        """
        训练时参数设置
        """
        self.opti_type = "Adam"
        self.opti_lr = 0.001                # 学习率
        self.opti_weight_decay = 0
        self.opti_amsgrad = True

        self.lr_scheduler_type = "WarmupPolyLR"
        self.lr_scheduler_warmup_epoch = 5

        self.trainer_seed = 2
        self.train_epochs = 1200
        self.train_log_iter = 10
        self.train_show_images_iter = 50
        # self.train_resume_checkpoint = "./output/DBNet_resnet18_FPN_DBHead/checkpoint/model_latest.pth"
        self.train_resume_checkpoint = ""
        self.train_finetune_checkpoint = ""
        self.train_output_dir = "output"

        self.post_processing_thresh = 0.3
        self.post_processing_box_thresh = 0.7
        self.post_processing_max_candidates = 1000
        self.post_processing_unclip_ratio = 1.5
        self.post_processing_type = "SegDetectorRepresenter"

        self.metric_type = "QuadMetric"
        self.metric_is_output_polygon = True


class Config(object):
    """
    图片数据增强参数设置
    """
    def __init__(self):
        super(Config, self).__init__()

        # 随机图片裁切参数设置-EastRandomCropData类参数
        self.random_crop_size = [1024, 1024]
        self.random_crop_max_tries = 50
        self.random_crop_keep_ratio = True
        self.random_crop_min_crop_side_ratio = 0.25     # 随机裁切的图片大小与原图的最小占比
        self.random_crop_require_original_image = False

        # MakeBorderMap类使用参数
        self.make_border_shrink_ratio = 0.4
        self.make_border_thresh_min = 0.3
        self.make_border_thresh_max = 0.7

        # MakeShrinkMap类使用参数
        self.make_shrink_shrink_ratio = 0.4             # 文本框收缩因子
        self.make_shrink_min_text_size = 8
        self.make_shrink_shrink_type = "pyclipper"      # 文本框收缩函数

        # ResizeShortSize类使用参数
        self.resize_short_size = 1024
        self.resize_text_polys = False                  # 是否对测试图片进行缩放

        # 训练参数设置
        self.train_batch_size = 4           # 训练Batch_size设置大小
        self.test_batch_size = 1            # 测试阶段Batch_size设置大小


def init_args():
    parser = argparse.ArgumentParser()
    # 数据集相关参数
    parser.add_argument("--datasets_path", default="{}/datasets".format(base_dir), help="数据集所在位置")
    parser.add_argument("--use_curved_text", default=False, help="是否使用弯曲文本")        # 暂不支持弯曲文本的检测
    parser.add_argument("--img_mode", default="RGB", help="图片格式必须是[RGB,BGR,GRAY]中的一种")
    parser.add_argument("--ignore_tags", default=['*', '###'], help="忽略的文本目标")

    parser.add_argument("--train_dataset_path", default="{}/datasets/train.txt".format(base_dir), help="训练数据集所在位置")
    parser.add_argument("--test_dataset_path", default="{}/datasets/test.txt".format(base_dir), help="测试数据集所在位置")
    parser.add_argument("--train_filter_keys", default=["img_path", "img_name", "text_polys", "text", "ignore_tags", "shape"], help="预处理后不需要的信息")
    parser.add_argument("--test_filter_keys", default=[], help="预处理后不需要的信息")

    # 网络结构设计
    parser.add_argument("--backbone_type", default="atros_resnet18", help="使用的backbone类型", type=str)
    parser.add_argument("--backbone_pretrained", default=True, help="是否使用预训练模型,默认使用")
    parser.add_argument("--backbone_in_channels", default=3, help="输入图片的格式,GRAY格式图片需设置为1")

    parser.add_argument("--neck_type", default="SPP", help="neck网络")
    parser.add_argument("--neck_inner_channels", default=256, help="neck网络channels")

    parser.add_argument("--head_type", default="DBHead", help="head头,即检测头")
    parser.add_argument("--head_k", default=50, help="可微分二值化计算参数")
    parser.add_argument("--head_smooth", default=False, help="是否对转置卷积结果进行平滑处理")

    # loss函数参数
    parser.add_argument("--loss_alpha", default=1.0, help="分类损失权重")
    parser.add_argument("--loss_beta", default=10.0, help="l1 loss权重")
    parser.add_argument("--loss_ohem", default=3, help="ohem参数")
    args = parser.parse_args()
    params = vars(args)
    return params


if __name__ == "__main__":
    params = init_args()
    print(params["datasets_path"])
