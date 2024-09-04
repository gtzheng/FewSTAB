import os

DATA_ROOT = "/experiments/few_shot_dataset"
MINIIMAGENET_PATH = os.path.join(DATA_ROOT, "miniImageNet--ravi")
TIEREDIMAGENET_PATH = os.path.join(DATA_ROOT, "tieredImageNet", "tiered_imagenet")
CIFAR100_PATH = os.path.join(DATA_ROOT, "CIFAR100")
CUBFEWSHOT_PATH = os.path.join(DATA_ROOT, "CUB_200_2011_FewShot")
CUB_PATH = os.path.join(DATA_ROOT, "CUB_birds_2010")
WEBCARICATURE_PATH = os.path.join(DATA_ROOT, "WebCaricature")
STANFORDDOG_PATH = os.path.join(DATA_ROOT, "StanfordDog")
STANFORDCAR_PATH = os.path.join(DATA_ROOT, "StanfordCar")


def get_data_folder(dataset):
    if dataset == "miniImagenet":
        data_folder = MINIIMAGENET_PATH
    elif dataset == "tieredImagenet":
        data_folder = TIEREDIMAGENET_PATH
    elif dataset == "CUBFewShot":
        data_folder = CUBFEWSHOT_PATH
    elif dataset == "StanfordCar":
        data_folder = STANFORDCAR_PATH
    elif dataset == "StanfordDog":
        data_folder = STANFORDDOG_PATH
    elif dataset == "CIFAR100":
        data_folder = CIFAR100_PATH
    elif dataset == "WebCaricature":
        data_folder = WEBCARICATURE_PATH
    else:
        data_folder = None
    return data_folder


chkpt_paths_mini_resnet12_5w1s = {
    "result_root": "./results",
    "data_root": "/experiments/few_shot_dataset/miniImageNet--ravi",
    "models": {
        "r2d2": "/experiments/few_shot_eval/R2D2/R2D2-miniImageNet--ravi-resnet12-5-1-Table2",
        "r2d2_spurious_vit-gpt2": "/experiments/few_shot_eval/R2D2/R2D2-miniImageNet--ravi-resnet12-5-1-spurious-att_vit-gpt2-Nov-06-2023-02-31-34",
        "r2d2_spurious_blip": "/experiments/few_shot_eval/R2D2/R2D2-miniImageNet--ravi-resnet12-5-1-spurious-att_blip-Nov-06-2023-02-34-11",
        "can": "/experiments/few_shot_eval/CAN/CAN-miniImageNet--ravi-resnet12-5-1-Table2",
        "leo": "/experiments/few_shot_eval/LEO/LEO-miniImageNet--ravi-resnet12-5-1-Table2",
        "renet": "/experiments/few_shot_eval/RENet/RENet-miniImageNet--ravi-resnet12-5-1-Table2",
        "rfs": "/experiments/few_shot_eval/RFS/RFS-simple-miniImageNet--ravi-resnet12-Table2",
        "skd": "/experiments/few_shot_eval/SKD/SKDModel-miniImageNet--ravi-resnet12-Gen1-Table2",
        "anil": "/experiments/few_shot_models/model_zoo/ANIL/ANIL-miniImageNet--ravi-resnet12-5-1-Table2",
        "boil": "/experiments/few_shot_eval/BOIL/BOIL-miniImageNet--ravi-resnet12-5-1-Table2",
        "dn4": "/experiments/few_shot_eval/DN4/DN4-miniImageNet--ravi-resnet12-5-1-Table2",
        "protonet": "/experiments/few_shot_eval/ProtoNet/ProtoNet-miniImageNet--ravi-resnet12-5-1-Table2",
        "protonet_spurious_vit-gpt2": "/experiments/few_shot_eval/ProtoNet/ProtoNet-miniImageNet--ravi-resnet12-5-1-spurious-att_vit-gpt2-Nov-06-2023-02-44-52",
        "protonet_spurious_blip": "/experiments/few_shot_eval/ProtoNet/ProtoNet-miniImageNet--ravi-resnet12-5-1-spurious-att_blip-Nov-06-2023-02-46-11",
        "baseline++": "/experiments/few_shot_eval/Baseline++/BaselinePlus-miniImageNet--ravi-resnet12-Table2",
    },
}

chkpt_paths_mini_resnet12_5w5s = {
    "result_root": "/results/FewSTAB/results",
    "data_root": "/experiments/few_shot_dataset/miniImageNet--ravi",
    "models": {
        "r2d2": "/experiments/few_shot_eval/R2D2/R2D2-miniImageNet--ravi-resnet12-5-5-Table2",
        "can": "/experiments/few_shot_eval/CAN/CAN-miniImageNet--ravi-resnet12-5-5-Table2",
        "leo": "/experiments/few_shot_eval/LEO/LEO-miniImageNet--ravi-resnet12-5-5-Table2",
        "renet": "/experiments/few_shot_eval/RENet/RENet-miniImageNet--ravi-resnet12-5-5-Table2",
        "rfs": "/experiments/few_shot_eval/RFS/RFS-simple-miniImageNet--ravi-resnet12-Table2",
        "skd": "/experiments/few_shot_eval/SKD/SKDModel-miniImageNet--ravi-resnet12-Gen1-Table2",
        "anil": "/experiments/few_shot_eval/ANIL/ANIL-miniImageNet--ravi-resnet12-5-5-Table2",
        "boil": "/experiments/few_shot_eval/BOIL/BOIL-miniImageNet--ravi-resnet12-5-5-Table2",
        "dn4": "/experiments/few_shot_eval/DN4/DN4-miniImageNet--ravi-resnet12-5-5-Table2",
        "protonet": "/experiments/few_shot_eval/ProtoNet/ProtoNet-miniImageNet--ravi-resnet12-5-5-Table2",
        "baseline++": "/experiments/few_shot_eval/Baseline++/BaselinePlus-miniImageNet--ravi-resnet12-Table2",
    },
}


chkpt_paths_tiered_resnet12_5w1s = {
    "result_root": "/results/FewSTAB/results",
    "data_root": "/experiments/few_shot_dataset/tieredImageNet/tiered_imagenet",
    "models": {
        "r2d2": "/experiments/few_shot_eval/R2D2/R2D2-tiered_imagenet-resnet12-5-1-Table2",
        "can": "/experiments/few_shot_eval/CAN/CAN-tiered_imagenet-resnet12-5-1-Table2",
        "leo": "/experiments/few_shot_eval/LEO/LEO-tiered_imagenet-resnet12-5-1-Table2",
        "renet": "/experiments/few_shot_eval/RENet/RENet-tiered_imagenet-resnet12-5-1-Reproduce",
        "rfs": "/experiments/few_shot_eval/RFS/RFS-simple-tiered_imagenet-resnet12-Table2",
        "skd": "/experiments/few_shot_eval/SKD/SKDModel-tiered_imagenet-resnet12-Gen1-Table2",
        "anil": "/experiments/few_shot_eval/ANIL/ANIL-tiered_imagenet-resnet12-5-1-Table2",
        "boil": "/experiments/few_shot_eval/BOIL/BOIL-tiered_imagenet-resnet12-5-1-Table2",
        "dn4": "/experiments/few_shot_eval/DN4/DN4-tiered_imagenet-resnet12-5-1-Table2",
        "protonet": "/experiments/few_shot_eval/ProtoNet/ProtoNet-tiered_imagenet-resnet12-5-1-Table2",
        "baseline++": "/experiments/few_shot_eval/Baseline++/BaselinePlus-tiered_imagenet-resnet12-Table2",
    },
}

chkpt_paths_tiered_resnet12_5w5s = {
    "result_root": "/results/FewSTAB/results",
    "data_root": "/experiments/few_shot_dataset/tieredImageNet/tiered_imagenet",
    "models": {
        "r2d2": "/experiments/few_shot_eval/R2D2/R2D2-tiered_imagenet-resnet12-5-5-Table2",
        "can": "/experiments/few_shot_eval/CAN/CAN-tiered_imagenet-resnet12-5-5-Table2",
        "leo": "/experiments/few_shot_eval/LEO/LEO-tiered_imagenet-resnet12-5-5-Table2",
        "renet": "/experiments/few_shot_eval/RENet/RENet-tiered_imagenet-resnet12-5-5-Reproduce",
        "rfs": "/experiments/few_shot_eval/RFS/RFS-simple-tiered_imagenet-resnet12-Table2",
        "skd": "/experiments/few_shot_eval/SKD/SKDModel-tiered_imagenet-resnet12-Gen1-Table2",
        "anil": "/experiments/few_shot_eval/ANIL/ANIL-tiered_imagenet-resnet12-5-5-Table2",
        "boil": "/experiments/few_shot_eval/BOIL/BOIL-tiered_imagenet-resnet12-5-5-Table2",
        "dn4": "/experiments/few_shot_eval/DN4/DN4-tiered_imagenet-resnet12-5-5-Table2",
        "protonet": "/experiments/few_shot_eval/ProtoNet/ProtoNet-tiered_imagenet-resnet12-5-5-Table2",
        "baseline++": "/experiments/few_shot_eval/Baseline++/BaselinePlus-tiered_imagenet-resnet12-Table2",
    },
}

chkpt_paths_cub_resnet12_5w1s = {
    "result_root": "/results/FewSTAB/results",
    "data_root": "/experiments/few_shot_dataset/CUB_200_2011_FewShot",
    "models": {
        "r2d2": "/experiments/few_shot_eval/R2D2/R2D2-CUB_200_2011_FewShot-resnet12-5-1",
        "can": "/experiments/few_shot_eval/CAN/CAN-CUB_200_2011_FewShot-resnet12-5-1",
        "leo": "/experiments/few_shot_eval/LEO/LEO-CUB_200_2011_FewShot-resnet12-5-1",
        "renet": "/experiments/few_shot_eval/RENet/RENet-CUB_200_2011_FewShot-resnet12-5-1",
        "rfs": "/experiments/few_shot_eval/RFS/RFSModel-CUB_200_2011_FewShot-resnet12-5-1",
        "anil": "/experiments/few_shot_eval/ANIL/ANIL-CUB_200_2011_FewShot-resnet12-5-1",
        "boil": "/experiments/few_shot_eval/BOIL/BOIL-CUB_200_2011_FewShot-resnet12-5-1",
        "dn4": "/experiments/few_shot_eval/DN4/DN4-CUB_200_2011_FewShot-resnet12-5-1",
        "protonet": "/experiments/few_shot_eval/ProtoNet/ProtoNet-CUB_200_2011_FewShot-resnet12-5-1",
        "baseline++": "/experiments/few_shot_eval/Baseline++/BaselinePlus-CUB_200_2011_FewShot-resnet12-5-1",
    },
}

chkpt_paths_cub_resnet12_5w5s = {
    "result_root": "/results/FewSTAB/results",
    "data_root": "/experiments/few_shot_dataset/CUB_200_2011_FewShot",
    "models": {
        "r2d2": "/experiments/few_shot_eval/R2D2/R2D2-CUB_200_2011_FewShot-resnet12-5-5",
        "can": "/experiments/few_shot_eval/CAN/CAN-CUB_200_2011_FewShot-resnet12-5-5",
        "leo": "/experiments/few_shot_eval/LEO/LEO-CUB_200_2011_FewShot-resnet12-5-5",
        "renet": "/experiments/few_shot_eval/RENet/RENet-CUB_200_2011_FewShot-resnet12-5-5",
        "rfs": "/experiments/few_shot_eval/RFS/RFSModel-CUB_200_2011_FewShot-resnet12-5-5",
        "anil": "/experiments/few_shot_eval/ANIL/ANIL-CUB_200_2011_FewShot-resnet12-5-5",
        "boil": "/experiments/few_shot_eval/BOIL/BOIL-CUB_200_2011_FewShot-resnet12-5-5",
        "dn4": "/experiments/few_shot_eval/DN4/DN4-CUB_200_2011_FewShot-resnet12-5-5",
        "protonet": "/experiments/few_shot_eval/ProtoNet/ProtoNet-CUB_200_2011_FewShot-resnet12-5-5",
        "baseline++": "/experiments/few_shot_eval/Baseline++/BaselinePlus-CUB_200_2011_FewShot-resnet12-5-5",
    },
}


def get_test_config(dataset, train_shot, caption_model):
    if dataset == "miniImagenet":
        if train_shot == 1:
            config = chkpt_paths_mini_resnet12_5w1s
        elif train_shot == 5:
            config = chkpt_paths_mini_resnet12_5w5s

        if caption_model == "vit-gpt2":
            config[
                "attribute_path_train"
            ] = "/experiments/few_shot_dataset/miniImageNet--ravi/train_vit-gpt2_attribute_embeds.pickle"
            config[
                "attribute_path_test"
            ] = "/experiments/few_shot_dataset/miniImageNet--ravi/test_vit-gpt2_attribute_embeds.pickle"
        elif caption_model == "blip":
            config[
                "attribute_path_train"
            ] = "/experiments/few_shot_dataset/miniImageNet--ravi/train_blip_attribute_embeds.pickle"
            config[
                "attribute_path_test"
            ] = "/experiments/few_shot_dataset/miniImageNet--ravi/test_blip_attribute_embeds.pickle"

    elif dataset == "tieredImagenet":
        if train_shot == 1:
            config = chkpt_paths_tiered_resnet12_5w1s
        elif train_shot == 5:
            config = chkpt_paths_tiered_resnet12_5w5s

        if caption_model == "vit-gpt2":
            config[
                "attribute_path_train"
            ] = "/experiments/few_shot_dataset/tieredImageNet/tiered_imagenet/train_vit-gpt2_attribute_embeds.pickle"
            config[
                "attribute_path_test"
            ] = "/experiments/few_shot_dataset/tieredImageNet/tiered_imagenet/test_vit-gpt2_attribute_embeds.pickle"
        elif caption_model == "blip":
            config[
                "attribute_path_train"
            ] = "/experiments/few_shot_dataset/tieredImageNet/tiered_imagenet/train_blip_attribute_embeds.pickle"
            config[
                "attribute_path_test"
            ] = "/experiments/few_shot_dataset/tieredImageNet/tiered_imagenet/test_blip_attribute_embeds.pickle"

    elif dataset == "CUBFewShot":
        if train_shot == 1:
            config = chkpt_paths_cub_resnet12_5w1s
        elif train_shot == 5:
            config = chkpt_paths_cub_resnet12_5w5s

        if caption_model == "vit-gpt2":
            config[
                "attribute_path_train"
            ] = "/experiments/few_shot_dataset/CUB_200_2011_FewShot/train_vit-gpt2_attribute_embeds.pickle"
            config[
                "attribute_path_test"
            ] = "/experiments/few_shot_dataset/CUB_200_2011_FewShot/test_vit-gpt2_attribute_embeds.pickle"

        elif caption_model == "blip":
            config[
                "attribute_path_train"
            ] = "/experiments/few_shot_dataset/CUB_200_2011_FewShot/train_blip_attribute_embeds.pickle"
            config[
                "attribute_path_test"
            ] = "/experiments/few_shot_dataset/CUB_200_2011_FewShot/test_blip_attribute_embeds.pickle"

    else:
        raise ValueError("Not a supported dataset")

    return config
