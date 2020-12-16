# -*- coding: utf-8 -*-
# @Time    : 2020/12/07 15:04
# @Author  : liyujun
from torchvision import transforms
from config.config import init_args, Config, EvalConfig
import time
import torch
from tqdm.auto import tqdm
from models.model import DBNet
from data_loader.image_preprocess.process_image import ICDAR2015Dataset
from data_loader.image_preprocess.resize_short_size import ResizeShortSize
from post_processing import get_post_processing
from utils.ocr_metric import get_metric
from data_loader.image_preprocess import ICDARCollectFN
from torch.utils.data import DataLoader


def eval():
    config = Config()
    eval_config = EvalConfig()
    params = init_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查设备
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
    checkpoint = torch.load(eval_config.model_path, map_location=torch.device('cpu'))
    # model = DBNet(params=params)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.to(device)

    model = torch.load(eval_config.model_path_model)
    model.to(device)


    # 生成配置文件
    post_processing_arg_dict = {
        "thresh": eval_config.post_processing_thresh,
        "box_thresh": eval_config.post_processing_box_thresh,
        "max_candidates": eval_config.post_processing_max_candidates,
        "unclip_ratio": eval_config.post_processing_unclip_ratio  # from paper
    }
    post_processing_arg = {
        "args": post_processing_arg_dict,
        "type": eval_config.post_processing_type
    }

    metric_arg_dict = {
        "is_output_polygon": eval_config.metric_is_output_polygon
    }
    metric_arg = {
        "type": eval_config.metric_type,
        "args": metric_arg_dict
    }
    val_config = {
        "post_processing": post_processing_arg,
        "metric": metric_arg
    }

    post_process = get_post_processing(val_config['post_processing'])
    metric_cls = get_metric(val_config['metric'])

    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    raw_metrics = []
    total_frame = 0.0
    total_time = 0.0
    for i, batch in tqdm(enumerate(testing_data_batch), total=len(testing_data_batch), desc='test model'):
        with torch.no_grad():
            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
            start = time.time()
            preds = model(batch['img'])
            boxes, scores = post_process(batch, preds, is_output_polygon=metric_cls.is_output_polygon)
            total_frame += batch['img'].size()[0]
            total_time += time.time() - start
            raw_metric = metric_cls.validate_measure(batch, (boxes, scores))
            raw_metrics.append(raw_metric)
    metrics = metric_cls.gather_measure(raw_metrics)
    print('FPS:{}'.format(total_frame / total_time))
    return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg


if __name__ == '__main__':
    result = eval()
    print(result)
