import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch

import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main_worker(cfg):
    cfg = default_setup(cfg)
    print(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

    trainer.train()



def main():
    args = default_argument_parser().parse_args()

    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),

    )



    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    main()
