from config.config import init_args, Config, TrainConfig
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader.image_preprocess import ICDARCollectFN
from data_loader.image_preprocess.process_image import ICDAR2015Dataset
from data_loader.image_preprocess.iaa_augment import ImageAug
from data_loader.image_preprocess.random_crop_data import EastRandomCropData
from data_loader.image_preprocess.make_border_map import MakeBorderMap
from data_loader.image_preprocess.make_shrink_map import MakeShrinkMap
from data_loader.image_preprocess.resize_short_size import ResizeShortSize

from models.model import DBNet
from tools.losses.db_loss import DBLoss

from tools.train_helper import Train_Helper


from utils.ocr_metric import get_metric
from post_processing import get_post_processing


def train():
    config = Config()
    train_config = TrainConfig()
    params = init_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"  # 检查设备

    # 加载训练数据集
    train_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_dataset = ICDAR2015Dataset(params=params, mode="train", img_transform=transforms.Compose(
        [ImageAug(), EastRandomCropData(config), MakeBorderMap(config), MakeShrinkMap(config)]),
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])]))

    training_data_batch = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True,
                                     **train_kwargs)

    # 加载测试数据集
    icdar_collect_fn = ICDARCollectFN()  # 自定义Batch生成类
    test_kwargs = {'num_workers': 1, 'pin_memory': False,
                   "collate_fn": icdar_collect_fn} if torch.cuda.is_available() else {}

    test_dataset = ICDAR2015Dataset(params=params, mode="test", img_transform=transforms.Compose(
        [ResizeShortSize(config)]),
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                       std=[0.229, 0.224, 0.225])]))
    testing_data_batch = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=False,
                                    **test_kwargs)

    model = DBNet(params=params)
    criterion = DBLoss(params=params).to(device)

    # 生成配置文件
    post_processing_arg_dict = {
        "thresh": train_config.post_processing_thresh,
        "box_thresh": train_config.post_processing_box_thresh,
        "max_candidates": train_config.post_processing_max_candidates,
        "unclip_ratio": train_config.post_processing_unclip_ratio  # from paper
    }
    post_processing_arg = {
        "args": post_processing_arg_dict,
        "type": train_config.post_processing_type
    }

    metric_arg_dict = {
        "is_output_polygon": train_config.metric_is_output_polygon
    }
    metric_arg = {
        "type": train_config.metric_type,
        "args": metric_arg_dict
    }
    val_config = {
        "post_processing": post_processing_arg,
        "metric": metric_arg
    }

    post_p = get_post_processing(val_config['post_processing'])
    metric = get_metric(val_config['metric'])

    train_helper = Train_Helper(train_config=train_config,
                                model=model,
                                criterion=criterion,
                                train_loader=training_data_batch,
                                validate_loader=testing_data_batch,
                                metric_cls=metric,
                                post_process=post_p
                                )

    train_helper.train()


if __name__ == "__main__":
    train()
