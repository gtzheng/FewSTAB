# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test, utils
import argparse
from spurious.config import get_test_config


def main(rank, config, path):
    test = Test(rank, config, path)
    test.test_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="anil", help="name of the method")
    parser.add_argument(
        "--spurious", default=False, action="store_true", help="spurious or normal test"
    )
    parser.add_argument(
        "--average",
        default=False,
        action="store_true",
        help="spurious or normal test",
    )
    parser.add_argument(
        "--dataset", type=str, default="miniImagenet", help="test dataset"
    )
    parser.add_argument(
        "--caption_model",
        type=str,
        default="vit-gpt2",
        help="selec a vision-language model",
    )

    parser.add_argument("--test_shot", type=int, default=5, help="test shot")
    parser.add_argument("--train_shot", type=int, default=1, help="training shot")
    parser.add_argument("--test_episode", type=int, default=100, help="number of test tasks")

    args = parser.parse_args()
    expr_configs = get_test_config(args.dataset, args.train_shot, args.caption_model)

    VAR_DICT = {
        "test_epoch": 1,
        "device_ids": "0",
        "n_gpu": 1,
        "test_episode": args.test_episode,
        "episode_size": 1,
        "test_query": 15,
        "test_shot": args.test_shot,
        "test_way": 5,
    }
    PATH = expr_configs["models"][args.method]

    gpu = ",".join([str(i) for i in utils.get_free_gpu()[0 : VAR_DICT["n_gpu"]]])
    VAR_DICT["device_ids"] = gpu
    utils.set_gpu(gpu)

    config = Config(
        os.path.join(PATH, "config.yaml"), VAR_DICT, enable_console=False
    ).get_config_dict()
    config["train_shot"] = args.train_shot
    config["shot_num"] = args.test_shot
    config["way_num"] = config["test_way"]
    config["worst_group"] = not args.average
    config["attribute_path_train"] = expr_configs["attribute_path_train"]
    config["attribute_path_test"] = expr_configs["attribute_path_test"]

    config["spurious"] = args.spurious
    config["data_root"] = expr_configs["data_root"]
    config["spurious_mode"] = "offline"
    
    config["task_path"] = os.path.join(
        expr_configs["data_root"],
        f"{args.dataset}_spurious_test_{args.test_episode}E_{config['test_way']}w{args.test_shot}s_{args.caption_model}_test_attrs.pt",
    )
    if args.average:
        acc_type = "_average"
    else:
        acc_type = "_worst_group"
    os.makedirs(expr_configs["result_root"],exist_ok=True)
    if args.spurious:
        config["result_path"] = os.path.join(
            expr_configs["result_root"],
            f"{args.dataset}_spurious_test_{args.test_episode}E_{config['test_way']}w{args.test_shot}s_{args.caption_model}{acc_type}_test_attrs.txt",
        )
    else:
        config["result_path"] = os.path.join(
            expr_configs["result_root"],
            f"{args.dataset}_test_{config['test_way']}w{args.test_shot}s{acc_type}.txt",
        )
    config["workers"] = 4
    config["seed"] = 0

    if config["n_gpu"] > 1:
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config, PATH)
