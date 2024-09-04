from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import os
from tqdm import tqdm
from core import utils
import argparse

from torch.utils.data import Dataset, DataLoader


def collate_fn(data):
    """Transform a list of tuples into a batch

    Args:
        data (list[tuple[np.ndarray, int, np.ndarray, np.ndarray]]): a list of tuples sampled from a dataset

    Returns:
        tensor: a list of tensor data
    """
    # data = a list of tuples
    batch_size = len(data)
    batch1 = [data[i][0] for i in range(batch_size)]
    batch2 = [data[i][1] for i in range(batch_size)]
    batch3 = [data[i][2] for i in range(batch_size)]
    return batch1, batch2, batch3


class ImageData(Dataset):
    def __init__(self, img_folder, csv_path):
        lines = []
        with open(csv_path, "r") as f:
            for x in f.readlines()[1:]:
                lines.append(x.strip())
        self.lines = lines
        self.img_folder = img_folder
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        path = self.lines[idx]
        image_path = os.path.join(self.img_folder, "images", path.split(',')[0]) 
        temp = Image.open(image_path)
        if temp.mode != "RGB":
            temp = temp.convert(mode="RGB")
        image = temp.copy()
        temp.close()

        # with Image.open(image_path) as i_image:
        #     if i_image.mode != "RGB":
        #         i_image = i_image.convert(mode="RGB")
        image_name = os.path.split(image_path)[1]
        label = path.split(',')[1].strip()
        return image, label, image_name

class VITGPT2_CAPTIONING:
    def __init__(self, max_length=16, num_beams=4):
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        if torch.cuda.is_available():
            gpu = utils.get_free_gpu()[0]
            self.device = f"cuda:{gpu}"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()
        
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict_step(self, image_paths):
        images = []
        image_names = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            image_names.append(os.path.split(image_path)[1])
            images.append(i_image)
        
        with torch.no_grad():
            pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            msgs = []
            for i in range(len(preds)):
                msgs.append(f"{image_names[i]},{preds[i].strip()}")
        return msgs

    def get_img_captions(self, img_folder, split, batch_size=256):
        csv_path = os.path.join(img_folder, "{}.csv".format(split))
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} does not exist")
        lines = [x.strip() for x in open(csv_path, "r").readlines()][1:]
        save_path = os.path.join(img_folder, "{}_vit-gpt2_captions.csv".format(split))
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                caption_lines = f.readlines()
            if len(caption_lines) == len(lines):
                print(f"{save_path} have been generated")
                return
        count = 0
        timer = utils.Timer()
        with open(save_path, "w") as fout:
            while count < len(lines):
                sel = lines[count:min(count+batch_size,len(lines))]
                paths = [os.path.join(img_folder, "images", s.split(',')[0]) for s in sel]
                labels = [s.split(',')[1].strip() for s in sel]
                msgs = self.predict_step(paths)
                write_info = '\n'.join([f"{msgs[i]},{labels[i]}" for i in range(len(msgs))])
                fout.write(f"{write_info}\n")
                fout.flush()
                count += batch_size
                elapsed_time = timer.t()
                est_time = elapsed_time/count * len(lines)
                print(f"Progress: {split} {count/len(lines)*100:.2f}% {utils.time_str(elapsed_time)}/est:{utils.time_str(est_time)}   ", end="\r")


class BLIP_CAPTIONING:
    def __init__(self, max_length=16, num_beams=4):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

        if torch.cuda.is_available():
            gpu = utils.get_free_gpu()[0]
            self.device = f"cuda:{gpu}"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


    def predict_step(self, data, names):
        
        with torch.no_grad():
            text = "there"
            inputs = self.processor(data, [text]*len(data), return_tensors="pt").to(self.device)
            output_ids = self.model.generate(**inputs)
            preds = self.processor.batch_decode(output_ids, skip_special_tokens=True)

            msgs = []
            for i in range(len(preds)):
                msgs.append(f"{names[i]},{preds[i].strip()}")
        return msgs

    def get_img_captions(self, img_folder, split, batch_size=512):
        csv_path = os.path.join(img_folder, "{}.csv".format(split))
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} does not exist")
        lines = []
        with open(csv_path, "r") as f:
            for x in f.readlines()[1:]:
                lines.append(x.strip())

        save_path = os.path.join(img_folder, "{}_blip_captions_test.csv".format(split))
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                caption_lines = f.readlines()
            if len(caption_lines) == len(lines):
                print(f"{save_path} has been generated")
                return
        dataset = ImageData(img_folder, csv_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=12,collate_fn=collate_fn)
        timer = utils.Timer()
        count = 0
        with open(save_path, "w") as fout:
            for data, labels, names in dataloader:
                msgs = self.predict_step(data, names)
                write_info = '\n'.join([f"{msgs[i]},{labels[i]}" for i in range(len(msgs))])
                fout.write(f"{write_info}\n")
                fout.flush()
                count += batch_size
                elapsed_time = timer.t()
                est_time = elapsed_time / count * len(lines)
                print(f"Progress: {split} {count / len(lines) * 100:.2f}% {utils.time_str(elapsed_time)}/est:{utils.time_str(est_time)}   ", end="\r")    

