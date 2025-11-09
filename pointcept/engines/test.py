"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)


TESTERS = Registry("testers")



class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        #构建模型并加载权重。
        model = build_model(self.cfg.model)
        #调用 build_model 函数（通常是一个工具函数），根据配置 self.cfg.model 构建模型。
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        #计算模型中可训练参数的数量，并记录到日志中。
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        # 将模型移动到GPU上（model.cuda()）。
        # 使用create_ddp_model将模型包装为分布式数据并行模型（DDP）。
        # broadcast_buffers = False：不广播缓冲区。
        # find_unused_parameters = self.cfg.find_unused_parameters：根据配置决定是否查找未使用的参数。
        if self.cfg.weight is None:
            raise ValueError("self.cfg.weight is None. Please ensure the weight path is correctly set in the config.")

        if not isinstance(self.cfg.weight, (str, bytes, os.PathLike)):
            raise TypeError(
                f"self.cfg.weight should be a string, bytes, or os.PathLike, but got {type(self.cfg.weight)}")

        if os.path.isfile(self.cfg.weight):
            #检查权重文件是否存在    加载权重
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight,weights_only=False)
            #此处已修改
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            #使用 model.load_state_dict 加载权重。
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        #构建测试数据加载器
        test_dataset = build_dataset(self.cfg.data.test)
        #调用 build_dataset 函数（通常是一个工具函数），根据配置构建测试数据集
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
# class SemSegTester(TesterBase):
#     def test(self):
#         assert self.test_loader.batch_size == 1
#         logger = get_root_logger()
#         logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
#
#         batch_time = AverageMeter()
#         intersection_meter = AverageMeter()
#         union_meter = AverageMeter()
#         target_meter = AverageMeter()
#         self.model.eval()
#
#         save_path = os.path.join(self.cfg.save_path, "result")
#         make_dirs(save_path)
#         # create submit folder only on main process
#         if (
#             self.cfg.data.test.type == "ScanNetDataset"
#             or self.cfg.data.test.type == "ScanNet200Dataset"
#             or self.cfg.data.test.type == "ScanNetPPDataset"
#         ) and comm.is_main_process():
#             make_dirs(os.path.join(save_path, "submit"))
#         elif (
#             self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
#         ):
#             make_dirs(os.path.join(save_path, "submit"))
#         elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
#             import json
#
#             make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
#             make_dirs(os.path.join(save_path, "submit", "test"))
#             submission = dict(
#                 meta=dict(
#                     use_camera=False,
#                     use_lidar=True,
#                     use_radar=False,
#                     use_map=False,
#                     use_external=False,
#                 )
#             )
#             with open(
#                 os.path.join(save_path, "submit", "test", "submission.json"), "w"
#             ) as f:
#                 json.dump(submission, f, indent=4)
#         comm.synchronize()
#         record = {}
#         # fragment inference
#         for idx, data_dict in enumerate(self.test_loader):
#             end = time.time()
#             data_dict = data_dict[0]  # current assume batch size is 1
#             fragment_list = data_dict.pop("fragment_list")
#             segment = data_dict.pop("segment")
#             data_name = data_dict.pop("name")
#             pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
#             if os.path.isfile(pred_save_path):
#                 logger.info(
#                     "{}/{}: {}, loaded pred and label.".format(
#                         idx + 1, len(self.test_loader), data_name
#                     )
#                 )
#                 pred = np.load(pred_save_path)
#                 if "origin_segment" in data_dict.keys():
#                     segment = data_dict["origin_segment"]
#             else:
#                 pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
#                 for i in range(len(fragment_list)):
#                     fragment_batch_size = 1
#                     s_i, e_i = i * fragment_batch_size, min(
#                         (i + 1) * fragment_batch_size, len(fragment_list)
#                     )
#                     input_dict = collate_fn(fragment_list[s_i:e_i])
#                     for key in input_dict.keys():
#                         if isinstance(input_dict[key], torch.Tensor):
#                             input_dict[key] = input_dict[key].cuda(non_blocking=True)
#                     idx_part = input_dict["index"]
#                     with torch.no_grad():
#                         pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
#                         pred_part = F.softmax(pred_part, -1)
#                         if self.cfg.empty_cache:
#                             torch.cuda.empty_cache()
#                         bs = 0
#                         for be in input_dict["offset"]:
#                             pred[idx_part[bs:be], :] += pred_part[bs:be]
#                             bs = be
#
#                     logger.info(
#                         "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
#                             idx + 1,
#                             len(self.test_loader),
#                             data_name=data_name,
#                             batch_idx=i,
#                             batch_num=len(fragment_list),
#                         )
#                     )
#                 if self.cfg.data.test.type == "ScanNetPPDataset":
#                     pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
#                 else:
#                     pred = pred.max(1)[1].data.cpu().numpy()
#                 if "origin_segment" in data_dict.keys():
#                     assert "inverse" in data_dict.keys()
#                     pred = pred[data_dict["inverse"]]
#                     segment = data_dict["origin_segment"]
#                 np.save(pred_save_path, pred)
#             if (
#                 self.cfg.data.test.type == "ScanNetDataset"
#                 or self.cfg.data.test.type == "ScanNet200Dataset"
#             ):
#                 np.savetxt(
#                     os.path.join(save_path, "submit", "{}.txt".format(data_name)),
#                     self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
#                     fmt="%d",
#                 )
#             elif self.cfg.data.test.type == "ScanNetPPDataset":
#                 np.savetxt(
#                     os.path.join(save_path, "submit", "{}.txt".format(data_name)),
#                     pred.astype(np.int32),
#                     delimiter=",",
#                     fmt="%d",
#                 )
#                 pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
#             elif self.cfg.data.test.type == "SemanticKITTIDataset":
#                 # 00_000000 -> 00, 000000
#                 sequence_name, frame_name = data_name.split("_")
#                 os.makedirs(
#                     os.path.join(
#                         save_path, "submit", "sequences", sequence_name, "predictions"
#                     ),
#                     exist_ok=True,
#                 )
#                 submit = pred.astype(np.uint32)
#                 submit = np.vectorize(
#                     self.test_loader.dataset.learning_map_inv.__getitem__
#                 )(submit).astype(np.uint32)
#                 submit.tofile(
#                     os.path.join(
#                         save_path,
#                         "submit",
#                         "sequences",
#                         sequence_name,
#                         "predictions",
#                         f"{frame_name}.label",
#                     )
#                 )
#             elif self.cfg.data.test.type == "NuScenesDataset":
#                 np.array(pred + 1).astype(np.uint8).tofile(
#                     os.path.join(
#                         save_path,
#                         "submit",
#                         "lidarseg",
#                         "test",
#                         "{}_lidarseg.bin".format(data_name),
#                     )
#                 )
#
#             intersection, union, target = intersection_and_union(
#                 pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
#             )
#             intersection_meter.update(intersection)
#             union_meter.update(union)
#             target_meter.update(target)
#             record[data_name] = dict(
#                 intersection=intersection, union=union, target=target
#             )
#
#             mask = union != 0
#             iou_class = intersection / (union + 1e-10)
#             iou = np.mean(iou_class[mask])
#             acc = sum(intersection) / (sum(target) + 1e-10)
#
#             m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
#             m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
#
#             batch_time.update(time.time() - end)
#             logger.info(
#                 "Test: {} [{}/{}]-{} "
#                 "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
#                 "Accuracy {acc:.4f} ({m_acc:.4f}) "
#                 "mIoU {iou:.4f} ({m_iou:.4f})".format(
#                     data_name,
#                     idx + 1,
#                     len(self.test_loader),
#                     segment.size,
#                     batch_time=batch_time,
#                     acc=acc,
#                     m_acc=m_acc,
#                     iou=iou,
#                     m_iou=m_iou,
#                 )
#             )
#
#         logger.info("Syncing ...")
#         comm.synchronize()
#         record_sync = comm.gather(record, dst=0)
#
#         if comm.is_main_process():
#             record = {}
#             for _ in range(len(record_sync)):
#                 r = record_sync.pop()
#                 record.update(r)
#                 del r
#             intersection = np.sum(
#                 [meters["intersection"] for _, meters in record.items()], axis=0
#             )
#             union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
#             target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
#
#             if self.cfg.data.test.type == "S3DISDataset":
#                 torch.save(
#                     dict(intersection=intersection, union=union, target=target),
#                     os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
#                 )
#
#             iou_class = intersection / (union + 1e-10)
#             accuracy_class = intersection / (target + 1e-10)
#             mIoU = np.mean(iou_class)
#             mAcc = np.mean(accuracy_class)
#             allAcc = sum(intersection) / (sum(target) + 1e-10)
#
#             logger.info(
#                 "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
#                     mIoU, mAcc, allAcc
#                 )
#             )
#             for i in range(self.cfg.data.num_classes):
#                 logger.info(
#                     "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
#                         idx=i,
#                         name=self.cfg.data.names[i],
#                         iou=iou_class[i],
#                         accuracy=accuracy_class[i],
#                     )
#                 )
#             logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

class SemSegTester(TesterBase):
    # 你的 collate_fn 定义在这里，但由于在类外也需要，我们暂时放在外面确保可用。
    # 实际项目中，如果 collate_fn 复杂，它可能在一个专门的文件中。
    # @staticmethod
    # def collate_fn(batch):
    #     return batch

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()  # 确保模型处于评估模式

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
                or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict_wrapper in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict_wrapper[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")  # <-- segment (真实标签) 的原始来源
            data_name = data_dict.pop("name")

            # !!! DEBUG POINT 1: 原始加载的 segment 形状
            print(
                f"DEBUG_SHAPE [{data_name}]: [1] Original segment shape from DataLoader: {segment.shape}, dtype: {segment.dtype}")

            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]  # <-- segment 可能在这里被替换
                # !!! DEBUG POINT 2a: 如果从文件加载 pred
                print(f"DEBUG_SHAPE [{data_name}]: [2a] Pred loaded from file: {pred.shape}, dtype: {pred.dtype}")
                print(
                    f"DEBUG_SHAPE [{data_name}]: [2a] Segment (after potential origin_segment): {segment.shape}, dtype: {segment.dtype}")

            else:
                # pred 初始化 (num_points, num_classes)
                # segment.size 是原始点云的总点数，用于初始化 pred 矩阵
                # 所以这里的 segment.size 应该和最终的 pred.shape[0] 匹配
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                print(f"DEBUG_SHAPE [{data_name}]: [2b] Initial pred (zeros) shape based on segment.size: {pred.shape}")

                for i in range(len(fragment_list)):
                    fragment_batch_size = 1  # 因为是 fragment 推理，通常是 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])  # collate_fn 在此处被调用

                    # !!! DEBUG POINT 2b.1: input_dict["index"] 的形状
                    if "index" in input_dict:
                        print(
                            f"DEBUG_SHAPE [{data_name}] fragment {i}: input_dict['index'] shape: {input_dict['index'].shape}, max index: {input_dict['index'].max()}")
                    else:
                        print(f"DEBUG_SHAPE [{data_name}] fragment {i}: input_dict does not contain 'index'")

                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]  # <-- 这个索引至关重要
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        # !!! DEBUG POINT 2b.2: 模型输出的 pred_part 形状
                        print(
                            f"DEBUG_SHAPE [{data_name}] fragment {i}/{len(fragment_list)}: Model output pred_part (seg_logits) shape: {pred_part.shape}")

                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            # pred[idx_part[bs:be], :] += pred_part[bs:be] # 累加到 pred 中
                            # !!! DEBUG POINT 2b.3: 累加到 pred 之前的 pred_part 片段形状
                            print(
                                f"DEBUG_SHAPE [{data_name}] fragment {i} pred_part slice shape (bs:{bs}, be:{be}): {pred_part[bs:be].shape}")

                            # !!! 检查这里是否会超出 pred 的索引范围，或者导致写入的位置不正确
                            # print(f"DEBUG_SHAPE [{data_name}] fragment {i} idx_part slice shape (bs:{bs}, be:{be}): {idx_part[bs:be].shape}, max_idx: {idx_part[bs:be].max()}")
                            # print(f"DEBUG_SHAPE [{data_name}] fragment {i} pred current shape before +=: {pred.shape}")
                            pred[idx_part[bs:be], :] += pred_part[bs:be]  # 累加
                            bs = be

                        logger.info(
                            "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                                idx + 1,
                                len(self.test_loader),
                                data_name=data_name,
                                batch_idx=i,
                                batch_num=len(fragment_list),
                            )
                        )
                # ... (处理 pred 以获得最终类别预测，即 pred.max(1)[1] 或 topk) ...
                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()  # <-- 这里的 pred 形状将是一维的

                # !!! DEBUG POINT 3: 最终的 pred 形状 (即将进入 intersection_and_union)
                print(
                    f"DEBUG_SHAPE [{data_name}]: [3] Final pred shape after argmax/topk: {pred.shape}, dtype: {pred.dtype}")

                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]  # <-- pred 可能在这里被重新索引/改变大小
                    segment = data_dict["origin_segment"]  # <-- segment 可能在这里被替换
                    # !!! DEBUG POINT 4: 经过 origin_segment/inverse 调整后的 pred 和 segment 形状
                    print(
                        f"DEBUG_SHAPE [{data_name}]: [4] Pred AFTER origin_segment/inverse adjustment: {pred.shape}, dtype: {pred.dtype}")
                    print(
                        f"DEBUG_SHAPE [{data_name}]: [4] Segment AFTER origin_segment/inverse adjustment: {segment.shape}, dtype: {segment.dtype}")

                np.save(pred_save_path, pred)  # 保存最终的 pred

            # ... (提交格式化代码，保持不变) ...

            # !!! DEBUG POINT 5: 最终要传递给 intersection_and_union 的形状
            print(
                f"DEBUG_SHAPE [{data_name}]: [5] FINAL SHAPE for intersection_and_union - 'pred': {pred.shape}, dtype: {pred.dtype}")
            print(
                f"DEBUG_SHAPE [{data_name}]: [5] FINAL SHAPE for intersection_and_union - 'segment': {segment.shape}, dtype: {segment.dtype}")

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,  # 注意这里依然使用 segment.size 打印
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
