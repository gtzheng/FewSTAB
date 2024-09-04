from spurious.config import *
from spurious.captioning import VITGPT2_CAPTIONING, BLIP_CAPTIONING
from spurious.get_attributes import main as get_attributes
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of a dataset")
    parser.add_argument("--model", type=str, default="vit-gpt2", help="image2text model")
    args = parser.parse_args()
    
    if args.model == "vit-gpt2":
        caption_model = VITGPT2_CAPTIONING()
    elif args.model == "blip":
        caption_model = BLIP_CAPTIONING()
    else:
        raise ValueError(f"Captioning model {args.model} not supported")
    
    data_folder = get_data_folder(args.dataset)
    
    for split in ["train", "val", "test"]:
        caption_model.get_img_captions(data_folder, split)
        get_attributes(args.dataset, split, args.model)