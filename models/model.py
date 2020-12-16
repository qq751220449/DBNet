from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head

import torch
import torch.nn as nn
import torch.nn.functional as F


class DBNet(nn.Module):
    def __init__(self, params):
        super(DBNet, self).__init__()
        self.backbone = build_backbone(backbone_type=params["backbone_type"], pretrained=params["backbone_pretrained"], in_channels=params["backbone_in_channels"])
        backbone_channel_out = self.backbone.get_channels()             # resnet网络输出通道列表
        self.neck = build_neck(neck_type=params["neck_type"], in_channels=backbone_channel_out, inner_channels=params["neck_inner_channels"])
        neck_channel_out = self.neck.out_channels
        self.head = build_head(head_type=params["head_type"], in_channels=neck_channel_out, smooth=params["head_smooth"], k=params["head_k"])
        self.name = f'{params["backbone_type"]}_{params["neck_type"]}_{params["head_type"]}'

    def forward(self, x):
        image_height, image_width = x.size()[2:]
        model_out = self.head(self.neck(self.backbone(x)))
        model_out = F.interpolate(model_out, size=(image_height, image_width), mode='bilinear', align_corners=True)
        return model_out


if __name__ == "__main__":
    from config.config import init_args, Config
    x = torch.zeros(2, 3, 1024, 1024).to("cuda")
    params = init_args()
    model = DBNet(params=params).to("cuda")
    # model.eval()
    out = model(x)
    print(out.shape)
    print(out[:, 0, :, :])
    print(out[:, 1, :, :])
    print(out[:, 2, :, :])

