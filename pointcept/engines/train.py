"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry


TRAINERS = Registry("trainers")


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        # 初始化一个空字典 self.comm_info，用于存储训练过程中需要跨方法传递的信息，例如当前迭代的输入数据、模型输出等。
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        # 设置最大训练 epoch 数为 cfg.eval_epoch
        self.best_metric_value = -torch.inf
        # 初始化最佳指标值为负无穷，用于记录训练过程中最佳的验证指标
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.csv"),
            file_mode="a" if cfg.resume else "w",
        )
        # 初始化日志记录器：
        # 日志文件路径为 cfg.save_path/train.log。
        # 如果 cfg.resume 为 True，则以追加模式（"a"）打开日志文件；否则以写入模式（"w"）打开。
        self.logger.info("=> Loading config ...")
        # 记录日志，表示正在加载配置
        self.cfg = cfg
        # 将传入的配置对象 cfg 赋值给类的成员变量 self.cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        # 记录保存路径
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        # 记录配置的详细内容。
        self.logger.info("=> Building model ...")
        # 记录日志，表示正在构建模型
        self.model = self.build_model()
        # 调用 build_model 方法构建模型，并将模型赋值给 self.model。
        # build_model 方法通常会根据配置 cfg 创建模型实例。
        self.logger.info("=> Building writer ...")
        # 记录日志，表示正在构建日志记录器
        self.writer = self.build_writer()
        # build_writer 方法通常会创建一个 SummaryWriter 实例，用于记录训练过程中的指标。
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # 当前 epoch 的索引，从 self.start_epoch 开始，到 self.max_epoch 结束。
                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                # 将模型设置为训练模式，启用 Dropout 和 BatchNorm 等层
                self.data_iterator = enumerate(self.train_loader)
                # 创建一个迭代器，用于遍历训练数据加载器中的每个 batch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # 遍历每个 batch 的数据，self.comm_info["iter"] 是当前迭代的索引，self.comm_info["input_dict"] 是当前 batch 的输入数据。
                    # => before_step
                    self.before_step()
                    # 调用 before_step 方法，执行每个迭代开始前的准备工作，例如记录日志、更新学习率等。
                    # => run_step
                    self.run_step()
                    # 调用 run_step 方法，执行单个迭代的训练步骤，包括前向传播、计算损失、反向传播和优化器更新。
                    # => after_step
                    self.after_step()
                # 调用 after_step 方法，执行每个迭代结束后的清理工作，例如记录日志、更新指标等
                # => after epoch
                self.after_epoch()
                # 调用 after_epoch 方法，执行每个 epoch 结束后的清理工作，例如保存模型、评估验证集性能等。
            # => after train
            self.after_train()
            # 调用 after_train 方法，执行训练结束后的清理工作，例如保存最终模型、关闭日志记录器等。

    def run_step(self):
        # 这段代码定义了 Trainer 类的 run_step 方法，负责执行单个训练迭代的核心逻辑。
        # 它包括数据准备、模型前向传播、损失计算、反向传播、优化器更新以及可选的混合精度训练
        input_dict = self.comm_info["input_dict"]
        # 获取当前迭代的输入数据字典 input_dict
        for key in input_dict.keys():
            # 遍历 input_dict 中的每个键
            if isinstance(input_dict[key], torch.Tensor):
                # 检查当前键对应的值是否为 torch.Tensor 类型。
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
                # 将张量移动到 GPU 上，使用 non_blocking=True 以非阻塞方式执行，提高效率
        #        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
        with torch.amp.autocast(device_type="cuda", enabled=self.cfg.enable_amp):
            # 使用 torch.amp.autocast 上下文管理器启用混合精度训练（AMP）
            # enabled=self.cfg.enable_amp 根据配置 self.cfg.enable_amp 决定是否启用混合精度训练。

            output_dict = self.model(input_dict)
            # 将输入数据 input_dict 传递给模型，获取模型的输出 output_dict
            loss = output_dict["loss"]
            # 从模型输出中提取损失值 loss
        self.optimizer.zero_grad()
        # 清零优化器中的梯度，防止梯度累积。
        if self.cfg.enable_amp:
            # 如果启用了混合精度训练：
            self.scaler.scale(loss).backward()
            # 使用 GradScaler 对损失进行缩放并执行反向传播
            self.scaler.step(self.optimizer)
            # 更新优化器，同时处理梯度缩放。

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            # 获取当前的梯度缩放因子。
            self.scaler.update()
            # 更新梯度缩放因子
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
                # 如果缩放因子没有变化，则调用学习率调度器的 step 方法。
                # 这是为了避免在梯度缩放因子过大时调用 optimizer.step，从而导致警告
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # 如果未启用混合精度训练：执行反向传播。更新优化器。调用学习率调度器的 step 方法。
        if self.cfg.empty_cache:
            # 如果配置中启用了 empty_cache：
            torch.cuda.empty_cache()
            # 释放未使用的 GPU 缓存内存，以减少内存占用。
        self.comm_info["model_output_dict"] = output_dict
        # 将模型的输出字典 output_dict 保存到 self.comm_info 中，以便后续使用

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader


# """
# Trainer
#
# Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
# Please cite our work if the code is helpful to you.
# """
#
# import os
# import sys
# import weakref
# import torch
# import torch.nn as nn
# import torch.utils.data
# from functools import partial
#
# if sys.version_info >= (3, 10):
#     from collections.abc import Iterator
# else:
#     from collections import Iterator
# from tensorboardX import SummaryWriter
#
# from .defaults import create_ddp_model, worker_init_fn
# from .hooks import HookBase, build_hooks
# import pointcept.utils.comm as comm
# from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
# from pointcept.models import build_model
# from pointcept.utils.logger import get_root_logger
# from pointcept.utils.optimizer import build_optimizer
# from pointcept.utils.scheduler import build_scheduler
# from pointcept.utils.events import EventStorage, ExceptionWriter
# from pointcept.utils.registry import Registry
# import matplotlib.pyplot as plt
#
# TRAINERS = Registry("trainers")
#
#
# class TrainerBase:
#     def __init__(self) -> None:
#         self.hooks = []
#         self.epoch = 0
#         self.start_epoch = 0
#         self.max_epoch = 0
#         self.max_iter = 0
#         self.comm_info = dict()
# #初始化一个空字典 self.comm_info，用于存储训练过程中需要跨方法传递的信息，例如当前迭代的输入数据、模型输出等。
#         self.data_iterator: Iterator = enumerate([])
#         self.storage: EventStorage
#         self.writer: SummaryWriter
#
#     def register_hooks(self, hooks) -> None:
#         hooks = build_hooks(hooks)
#         for h in hooks:
#             assert isinstance(h, HookBase)
#             # To avoid circular reference, hooks and trainer cannot own each other.
#             # This normally does not matter, but will cause memory leak if the
#             # involved objects contain __del__:
#             # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
#             h.trainer = weakref.proxy(self)
#         self.hooks.extend(hooks)
#
#     def train(self):
#         with EventStorage() as self.storage:
#             # => before train
#             self.before_train()
#             for self.epoch in range(self.start_epoch, self.max_epoch):
#                 # => before epoch
#                 self.before_epoch()
#                 # => run_epoch
#                 for (
#                     self.comm_info["iter"],
#                     self.comm_info["input_dict"],
#                 ) in self.data_iterator:
#                     # => before_step
#                     self.before_step()
#                     # => run_step
#                     self.run_step()
#                     # => after_step
#                     self.after_step()
#                 # => after epoch
#                 self.after_epoch()
#             # => after train
#             self.after_train()
#
#     def before_train(self):
#         for h in self.hooks:
#             h.before_train()
#
#     def before_epoch(self):
#         for h in self.hooks:
#             h.before_epoch()
#
#     def before_step(self):
#         for h in self.hooks:
#             h.before_step()
#
#     def run_step(self):
#         raise NotImplementedError
#
#     def after_step(self):
#         for h in self.hooks:
#             h.after_step()
#
#     def after_epoch(self):
#         for h in self.hooks:
#             h.after_epoch()
#         self.storage.reset_histories()
#
#     def after_train(self):
#         # Sync GPU before running train hooks
#         comm.synchronize()
#         for h in self.hooks:
#             h.after_train()
#         if comm.is_main_process():
#             self.writer.close()
#
#
# @TRAINERS.register_module("DefaultTrainer")
# class Trainer(TrainerBase):
#     def __init__(self, cfg):
#         super(Trainer, self).__init__()
#         self.epoch = 0
#         self.start_epoch = 0
#         self.max_epoch = cfg.eval_epoch
#         #设置最大训练 epoch 数为 cfg.eval_epoch
#         self.best_metric_value = -torch.inf
#         #初始化最佳指标值为负无穷，用于记录训练过程中最佳的验证指标
#         self.logger = get_root_logger(
#             log_file=os.path.join(cfg.save_path, "train.csv"),
#             file_mode="a" if cfg.resume else "w",
#         )
#         #初始化日志记录器：
#         #日志文件路径为 cfg.save_path/train.log。
#         #如果 cfg.resume 为 True，则以追加模式（"a"）打开日志文件；否则以写入模式（"w"）打开。
#         self.logger.info("=> Loading config ...")
#         #记录日志，表示正在加载配置
#         self.cfg = cfg
#         #将传入的配置对象 cfg 赋值给类的成员变量 self.cfg
#         self.logger.info(f"Save path: {cfg.save_path}")
#         #记录保存路径
#         self.logger.info(f"Config:\n{cfg.pretty_text}")
#         #记录配置的详细内容。
#         self.logger.info("=> Building model ...")
#         #记录日志，表示正在构建模型
#         self.model = self.build_model()
#         # 调用 build_model 方法构建模型，并将模型赋值给 self.model。
#         # build_model 方法通常会根据配置 cfg 创建模型实例。
#         self.logger.info("=> Building writer ...")
#         #记录日志，表示正在构建日志记录器
#         self.writer = self.build_writer()
#         #build_writer 方法通常会创建一个 SummaryWriter 实例，用于记录训练过程中的指标。
#         self.logger.info("=> Building train dataset & dataloader ...")
#         self.train_loader = self.build_train_loader()
#         self.logger.info("=> Building val dataset & dataloader ...")
#         self.val_loader = self.build_val_loader()
#         self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
#         self.optimizer = self.build_optimizer()
#         self.scheduler = self.build_scheduler()
#         self.scaler = self.build_scaler()
#         self.logger.info("=> Building hooks ...")
#         self.register_hooks(self.cfg.hooks)
#
#         # 新增：用于存储训练和验证指标的历史记录
#         self.train_metrics = {"loss": [], "mIoU": []}
#         self.val_metrics = {"loss": [], "mIoU": []}
#
#     def train(self):
#         with EventStorage() as self.storage, ExceptionWriter():
#             # => before train
#             self.before_train()
#             self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
#             for self.epoch in range(self.start_epoch, self.max_epoch):
#                 #当前 epoch 的索引，从 self.start_epoch 开始，到 self.max_epoch 结束。
#                 # => before epoch
#                 # TODO: optimize to iteration based
#                 if comm.get_world_size() > 1:
#                     self.train_loader.sampler.set_epoch(self.epoch)
#                 self.model.train()
#                 #将模型设置为训练模式，启用 Dropout 和 BatchNorm 等层
#                 self.data_iterator = enumerate(self.train_loader)
#                 #创建一个迭代器，用于遍历训练数据加载器中的每个 batch
#                 self.before_epoch()
#                 # => run_epoch
#                 for (
#                     self.comm_info["iter"],
#                     self.comm_info["input_dict"],
#                 ) in self.data_iterator:
#                 #遍历每个 batch 的数据，self.comm_info["iter"] 是当前迭代的索引，self.comm_info["input_dict"] 是当前 batch 的输入数据。
#                     # => before_step
#                     self.before_step()
#                 #调用 before_step 方法，执行每个迭代开始前的准备工作，例如记录日志、更新学习率等。
#                     # => run_step
#                     self.run_step()
#                 #调用 run_step 方法，执行单个迭代的训练步骤，包括前向传播、计算损失、反向传播和优化器更新。
#                     # => after_step
#                     self.after_step()
#                 #调用 after_step 方法，执行每个迭代结束后的清理工作，例如记录日志、更新指标等
#                 # => after epoch
#                 self.after_epoch()
#                 #调用 after_epoch 方法，执行每个 epoch 结束后的清理工作，例如保存模型、评估验证集性能等。
#             # => after train
#             self.after_train()
#             #调用 after_train 方法，执行训练结束后的清理工作，例如保存最终模型、关闭日志记录器等。
#
#     def run_step(self):
#         #这段代码定义了 Trainer 类的 run_step 方法，负责执行单个训练迭代的核心逻辑。
#         # 它包括数据准备、模型前向传播、损失计算、反向传播、优化器更新以及可选的混合精度训练
#         input_dict = self.comm_info["input_dict"]
#         #获取当前迭代的输入数据字典 input_dict
#         for key in input_dict.keys():
#             #遍历 input_dict 中的每个键
#             if isinstance(input_dict[key], torch.Tensor):
#                 #检查当前键对应的值是否为 torch.Tensor 类型。
#                 input_dict[key] = input_dict[key].cuda(non_blocking=True)
#                 #将张量移动到 GPU 上，使用 non_blocking=True 以非阻塞方式执行，提高效率
# #        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
#         with torch.amp.autocast(device_type='cuda', enabled=self.cfg.enable_amp):
#             #使用 torch.amp.autocast 上下文管理器启用混合精度训练（AMP）
#             #enabled=self.cfg.enable_amp 根据配置 self.cfg.enable_amp 决定是否启用混合精度训练。
#
#             output_dict = self.model(input_dict)
#             #将输入数据 input_dict 传递给模型，获取模型的输出 output_dict
#             loss = output_dict["loss"]
#             #从模型输出中提取损失值 loss
#         self.optimizer.zero_grad()
#         #清零优化器中的梯度，防止梯度累积。
#         if self.cfg.enable_amp:
#             #如果启用了混合精度训练：
#             self.scaler.scale(loss).backward()
#             #使用 GradScaler 对损失进行缩放并执行反向传播
#             self.scaler.step(self.optimizer)
#             #更新优化器，同时处理梯度缩放。
#
#             # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
#             # Fix torch warning scheduler step before optimizer step.
#             scaler = self.scaler.get_scale()
#             #获取当前的梯度缩放因子。
#             self.scaler.update()
#             #更新梯度缩放因子
#             if scaler <= self.scaler.get_scale():
#                 self.scheduler.step()
#                 #如果缩放因子没有变化，则调用学习率调度器的 step 方法。
#                 #这是为了避免在梯度缩放因子过大时调用 optimizer.step，从而导致警告
#         else:
#             loss.backward()
#             self.optimizer.step()
#             self.scheduler.step()
#             #如果未启用混合精度训练：执行反向传播。更新优化器。调用学习率调度器的 step 方法。
#         if self.cfg.empty_cache:
#             #如果配置中启用了 empty_cache：
#             torch.cuda.empty_cache()
#             #释放未使用的 GPU 缓存内存，以减少内存占用。
#         self.comm_info["model_output_dict"] = output_dict
#         #将模型的输出字典 output_dict 保存到 self.comm_info 中，以便后续使用
#
#     def after_epoch(self):
#         for h in self.hooks:
#             h.after_epoch()
#         self.storage.reset_histories()
#         if self.cfg.empty_cache_per_epoch:
#             torch.cuda.empty_cache()
#
#             # 新增：在每个 epoch 结束后绘制训练曲线
#             self._plot_metrics()
#
#     def _plot_metrics(self):
#         # 绘制训练和验证的损失曲线
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(self.train_metrics["loss"], label="Train Loss")
#         if self.val_metrics["loss"]:
#             plt.plot(self.val_metrics["loss"], label="Val Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.title("Loss Curve")
#         plt.legend()
#
#         # 绘制训练和验证的 mIoU 曲线
#         plt.subplot(1, 2, 2)
#         plt.plot(self.train_metrics["mIoU"], label="Train mIoU")
#         if self.val_metrics["mIoU"]:
#             plt.plot(self.val_metrics["mIoU"], label="Val mIoU")
#         plt.xlabel("Epoch")
#         plt.ylabel("mIoU")
#         plt.title("mIoU Curve")
#         plt.legend()
#
#         # 保存图表
#         save_path = os.path.join(self.cfg.save_path, "training_curves.png")
#         plt.savefig(save_path)
#         plt.close()
#         self.logger.info(f"Training curves saved to {save_path}")
#
#         # 新增：在评估时更新验证指标
#
#     def after_step(self):
#         for h in self.hooks:
#             h.after_step()
#         # 假设 loss 和 IoU 已经计算并存储在 self.storage 中
#         # 这里只是一个示例，实际实现可能需要根据具体任务调整
#         if "loss" in self.storage.histories():
#             current_loss = self.storage.histories()["loss"].avg
#             self.train_metrics["loss"].append(current_loss)
#         if "mIoU" in self.storage.histories():
#             current_mIoU = self.storage.histories()["mIoU"].avg
#             self.train_metrics["mIoU"].append(current_mIoU)
#
#         # 新增：在验证阶段更新验证指标
#
#     def evaluate(self):
#         # 这里只是一个示例，实际验证逻辑需要根据具体任务实现
#         self.model.eval()
#         val_loss = 0.0
#         val_mIoU = 0.0
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 # 假设验证数据的前向传播和指标计算
#                 input_dict = batch
#                 for key in input_dict.keys():
#                     if isinstance(input_dict[key], torch.Tensor):
#                         input_dict[key] = input_dict[key].cuda(non_blocking=True)
#                 output_dict = self.model(input_dict)
#                 loss = output_dict["loss"]
#                 mIoU = output_dict.get("mIoU", 0.0)
#                 val_loss += loss.item()
#                 val_mIoU += mIoU
#         val_loss /= len(self.val_loader)
#         val_mIoU /= len(self.val_loader)
#         self.val_metrics["loss"].append(val_loss)
#         self.val_metrics["mIoU"].append(val_mIoU)
#         self.logger.info(f"Validation - Epoch: {self.epoch}, Loss: {val_loss:.4f}, mIoU: {val_mIoU:.4f}")
#
#
#     def build_model(self):
#         model = build_model(self.cfg.model)
#         if self.cfg.sync_bn:
#             model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         # logger.info(f"Model: \n{self.model}")
#         self.logger.info(f"Num params: {n_parameters}")
#         model = create_ddp_model(
#             model.cuda(),
#             broadcast_buffers=False,
#             find_unused_parameters=self.cfg.find_unused_parameters,
#         )
#         return model
#
#     def build_writer(self):
#         writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
#         self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
#         return writer
#
#     def build_train_loader(self):
#         train_data = build_dataset(self.cfg.data.train)
#
#         if comm.get_world_size() > 1:
#             train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
#         else:
#             train_sampler = None
#
#         init_fn = (
#             partial(
#                 worker_init_fn,
#                 num_workers=self.cfg.num_worker_per_gpu,
#                 rank=comm.get_rank(),
#                 seed=self.cfg.seed,
#             )
#             if self.cfg.seed is not None
#             else None
#         )
#
#         train_loader = torch.utils.data.DataLoader(
#             train_data,
#             batch_size=self.cfg.batch_size_per_gpu,
#             shuffle=(train_sampler is None),
#             num_workers=self.cfg.num_worker_per_gpu,
#             sampler=train_sampler,
#             collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
#             pin_memory=True,
#             worker_init_fn=init_fn,
#             drop_last=True,
#             persistent_workers=True,
#         )
#         return train_loader
#
#     def build_val_loader(self):
#         val_loader = None
#         if self.cfg.evaluate:
#             val_data = build_dataset(self.cfg.data.val)
#             if comm.get_world_size() > 1:
#                 val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
#             else:
#                 val_sampler = None
#             val_loader = torch.utils.data.DataLoader(
#                 val_data,
#                 batch_size=self.cfg.batch_size_val_per_gpu,
#                 shuffle=False,
#                 num_workers=self.cfg.num_worker_per_gpu,
#                 pin_memory=True,
#                 sampler=val_sampler,
#                 collate_fn=collate_fn,
#             )
#         return val_loader
#
#     def build_optimizer(self):
#         return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)
#
#     def build_scheduler(self):
#         assert hasattr(self, "optimizer")
#         assert hasattr(self, "train_loader")
#         self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
#         return build_scheduler(self.cfg.scheduler, self.optimizer)
#
#     def build_scaler(self):
#         scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
#         return scaler
#
#
# @TRAINERS.register_module("MultiDatasetTrainer")
# class MultiDatasetTrainer(Trainer):
#     def build_train_loader(self):
#         from pointcept.datasets import MultiDatasetDataloader
#
#         train_data = build_dataset(self.cfg.data.train)
#         train_loader = MultiDatasetDataloader(
#             train_data,
#             self.cfg.batch_size_per_gpu,
#             self.cfg.num_worker_per_gpu,
#             self.cfg.mix_prob,
#             self.cfg.seed,
#         )
#         self.comm_info["iter_per_epoch"] = len(train_loader)
#         return train_loader
