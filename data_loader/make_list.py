"""
@time: 2020/11/06 11:36
@author: 蓝天月影
@contact: qq751220449@126.com
"""

import re
import os
import sys
from sklearn.model_selection import train_test_split


# root目录设置
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
sys.path.append(base_dir)  # 设置项目根目录

from config.config import init_args


def make_list(params):
    datasets_root_path = params["datasets_path"]
    gt_path = os.path.abspath(os.path.join(datasets_root_path, "gt/"))
    img_path = os.path.abspath(os.path.join(datasets_root_path, "img/"))
    assert os.path.exists(gt_path) and os.path.exists(img_path)     # 检查文件路径是否存在
    img_list = os.listdir(img_path)         # img列表文件
    img_gt_list = []
    for img_name in img_list:
        gt_single_path = os.path.abspath(os.path.join(gt_path, str(img_name.split(".")[0]) + ".txt"))
        assert os.path.exists(gt_single_path)           # 判断对应的gt路径是否存在
        if not params["use_curved_text"]:
            # 不使用弯曲文本,暂不支持弯曲文本的处理
            re_result = re.match("^[0-9]+$", str(img_name.split(".")[0]))
            if re_result is not None:
                img_gt_list.append((os.path.abspath(os.path.join(img_path, img_name)), gt_single_path))
            else:
                lines_list = []     # 判断每一行文本
                with open(gt_single_path, "r", encoding="utf-8") as fd:
                    for line in fd.readlines():
                        if len(line.strip().split(",")) == 9:
                            lines_list.append("1")
                        else:
                            lines_list.append("0")
                    fd.close()
                if "0" not in lines_list:
                    img_gt_list.append((os.path.abspath(os.path.join(img_path, img_name)), gt_single_path))
        else:
            img_gt_list.append((os.path.abspath(os.path.join(img_path, img_name)), gt_single_path))
    datasets_len = len(img_gt_list)                                                     # 训练样本长度
    params["datasets_len"] = datasets_len
    x_train, x_test = train_test_split(img_gt_list, test_size=0.2, random_state=0)     # 分割训练样本与测试样本
    print("len of train dataset is {}, len of test dataset is {}".format(len(x_train), len(x_test)))
    # 将训练数据集写入txt文件中
    with open(os.path.abspath(os.path.join(datasets_root_path, "train.txt")), mode="w", encoding="utf-8") as fd_train:
        # 生成数据集路径
        for train_sample in x_train:
            fd_train.write(train_sample[0] + "\t" + train_sample[1])
            fd_train.write("\n")
        fd_train.close()
    # 将测试数据集写入txt文件中
    with open(os.path.abspath(os.path.join(datasets_root_path, "test.txt")), mode="w", encoding="utf-8") as fd_test:
        # 生成数据集路径
        for test_sample in x_test:
            fd_test.write(test_sample[0] + "\t" + test_sample[1])
            fd_test.write("\n")
        fd_test.close()


def dataset_preprocess(params):
    datasets_root_path = params["datasets_path"]
    gt_path = os.path.abspath(os.path.join(datasets_root_path, "gt/"))
    gt_list = os.listdir(gt_path)  # img列表文件
    for gt_text in gt_list:
        try:
            new_name = gt_text.split("gt_")[1]
            src = os.path.abspath(os.path.join(gt_path, gt_text))
            dst = os.path.abspath(os.path.join(gt_path, new_name))
            os.rename(src, dst)
        except:
            pass


if __name__ == "__main__":
    params = init_args()
    make_list(params)
    # dataset_preprocess(params)
