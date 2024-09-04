# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer, utils


# setfacl -m u:username:rwx myfolder
def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    config = Config("./config/spurious/proto_mini.yaml").get_config_dict()
    gpu = ",".join([str(i) for i in utils.get_free_gpu()[0 : config["n_gpu"]]])
    utils.set_gpu(gpu)
    config["device_ids"] = gpu
    if config["n_gpu"] > 1:
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
