import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
#用于实现点云分割任务的模型
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
            #num_classes：目标分割的类别数量
        backbone_out_channels,
            #backbone_out_channels：主干网络（backbone）输出的特征通道数
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        #如果 num_classes > 0，则初始化一个线性层（nn.Linear），将主干网络的输出特征映射到类别数量的分割 logits。
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)


    def forward(self, input_dict):
        point = Point(input_dict)
        print("point.feat0:", point.feat.shape)
    #将输入数据封装为一个 Point 类型的对象。Point 类可能是一个自定义类，用于表示点云数据及其相关特征。
        point = self.backbone(point)
    #主干网络（Backbone），负责提取输入数据的特征 主干网络的输出可能是一个 Point 对象或特征张量
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
        #检查主干网络的输出是否为 Point 类型。
            feat = point.feat
        #如果是 Point 类型，则提取其 feat 属性作为特征张量。
        else:
            feat = point
            #如果不是 Point 类型，则直接将主干网络的输出作为特征张量
        seg_logits = self.seg_head(feat)
        #分割头（Segmentation Head），负责将特征张量转换为分割 logits。
        #seg_logits 是分割头的输出，表示每个点属于不同类别的预测分数

        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
