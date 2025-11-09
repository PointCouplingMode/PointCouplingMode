"""
Default training/testing logic

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import argparse
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


import pointcept.utils.comm as comm
from pointcept.utils.env import get_random_seed, set_seed
from pointcept.utils.config import Config, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # parser.add_argument('--weight', type=str, required=True ,default=None,
    #                    help='Path to model weights')  # 使用required强制参数‌:ml-citation{ref="3" data="citationList"}

    parser.add_argument(
        "--config-file",
        default="/home/oem/Pycharm_Pytorch_Projects/PointTransformer/Pointcept-main/configs/s3dis/s3dis_mamba_change_MLP.py/",
        metavar="FILE", help="path to config file")
        # ("--config-file", default="/home/oem/Pycharm_Pytorch_Projects/PointTransformer/Pointcept-main/configs/scannet/semseg-pt-v3m1-0-base.py/", metavar="FILE", help="path to config file")

    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    parser.add_argument("--vis_result",
                        action="store_true",
                        default=False,
                        help="Enable visualization of prediction results")
    return parser


def default_config_parser(file_path, options):
    file_path = file_path.rstrip(os.sep)
    #移除路径末尾的系统路径分隔符（如 / 或 \），以避免后续处理中出现多余的分隔符
    if os.path.isabs(file_path):
        #检查 file_path 是否是绝对路径
        if os.path.isfile(file_path):
            return Config.fromfile(file_path)
        #如果路径指向一个文件（文件存在），则直接使用 Config.fromfile 方法加载配置文件并返回

        possible_exts = [".py", ".yaml", ".yml"]
        for ext in possible_exts:
            full_path = f"{file_path}{ext}"
            if os.path.isfile(full_path):
                return Config.fromfile(full_path)
            #如果路径不指向文件，则尝试在路径后添加可能的扩展名（.py, .yaml, .yml），并检查是否存在文件

        if os.path.isdir(file_path):
            default_files = ["config.py", "config.yaml", "config.yml"]
            for fname in default_files:
                full_path = os.path.join(file_path, fname)
                if os.path.isfile(full_path):
                    return Config.fromfile(full_path)
         #如果路径指向一个目录，则尝试在目录中查找默认的配置文件（如 config.py, config.yaml, config.yml）

        raise FileNotFoundError(
            f"绝对路径配置不存在（需检查路径/扩展名）:\n"
            f"原始输入: {file_path}\n"
            f"尝试补充扩展名: {[f'{file_path}{ext}' for ext in possible_exts]}"
        )


    if "/" not in file_path:
        raise ValueError(f"协议路径需为 dataset/model-exp，实际输入: {file_path}")
    #如果路径中不包含 /，则认为输入的路径格式不符合协议要求（dataset/model-exp），并抛出一个 ValueError

    dataset, model_exp = file_path.split("/", 1)
    #数据集名称 模型名称
    config_path = os.path.join("configs", dataset, f"{model_exp}.py")
    #构造配置文件路径为 configs/dataset/model_exp.py
    print(config_path)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"协议路径配置文件不存在: {config_path}")
    #检查构造的配置文件路径是否存在
    return Config.fromfile(config_path)

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch
    #计算训练数据的循环次数，并将其赋值给 cfg.data.train.loop

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    #确保保存路径下的 model 子目录存在。如果目录不存在，则创建它。exist_ok=True 表示如果目录已存在，不会抛出错误
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    #获取当前集群的"世界大小"（即参与计算的节点或GPU的数量，表示为 world_size）。
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    #如果用户没有为 num_worker 设置值，则根据机器的CPU核心数（mp.cpu_count()）来设置线程数。num_worker 通常用于指定数据加载的工作线程数。
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    #将总线程数（num_worker）平均分配到每个 GPU 上。在多GPU环境中，每个 GPU 都需要独立的数据加载线程，因此需要根据 world_size 来计算每个 GPU 的线程数
    assert cfg.batch_size % world_size == 0
    #确保批量大小（batch_size）必须能够被 world_size 整除
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    #如果 batch_size_val 和 batch_size_test 值不为 None，则检查它们是否也能被 world_size 整除
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    #计算每个 GPU 上的批量大小（batch_size_per_gpu）、验证批量大小（batch_size_val_per_gpu）和测试批量大小（batch_size_test_per_gpu）。
    # #如果原始的批量大小未设置，则默认为每个 GPU 处理 1 个样本。
    # update data loop
    assert cfg.epoch % cfg.eval_epoch == 0
    #确保训练周期（cfg.epoch）能够被评估周期（cfg.eval_epoch）整除。这是为了确保在训练过程中，评估操作可以在合适的间隔点触发。
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed * cfg.num_worker_per_gpu + rank
    #根据配置文件中的种子值（cfg.seed）和当前的秩（rank）计算一个随机种子
    set_seed(seed)
    return cfg
