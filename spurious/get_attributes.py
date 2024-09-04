import spacy
from collections import Counter
from spacy.tokenizer import Tokenizer
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from .config import get_data_folder


def to_singular(nlp, text):
    doc = nlp(text)
    if len(doc) == 1:
        return doc[0].lemma_
    else:
        return doc[:-1].text + doc[-2].whitespace_ + doc[-1].lemma_


def get_adj_pairs(doc):
    adj_set = set()
    for chunk in doc.noun_chunks:
        adj = []
        split = False
        noun = ""
        for tok in chunk:
            if tok.pos_ == "ADJ":
                adj.append(f"{tok.text}:adj")
        for a in adj:
            adj_set.add(a)

    return list(adj_set)


def get_nouns(nlp, doc):
    nouns = []
    noun_set = set()
    for tok in doc:
        if tok.dep_ == "compound":
            comp_str = doc[tok.i : tok.head.i + 1]
            comp_str = to_singular(nlp, comp_str.text)
            for n in comp_str.split(" "):
                noun_set.add(f"{n}:noun")
            nouns.append(f"{comp_str}:noun")
    for tok in doc:
        if tok.pos_ == "NOUN":
            text = tok.text
            if tok.tag_ in {"NNS", "NNPS"}:
                text = tok.lemma_
            if text not in noun_set:
                nouns.append(f"{text}:noun")
    return nouns


def extract_attributes(nlp, texts):
    docs = nlp.pipe(texts)
    attributes = {}
    for doc in docs:
        adjs = get_adj_pairs(doc)
        nouns = get_nouns(nlp, doc)
        for a in adjs:
            attributes[a] = "adj"
        for n in nouns:
            attributes[n] = "noun"
    return attributes


def get_img_embed(nlp, word_list):
    attributes_count = {}
    attributes_arr = []
    # calculate attribute frequency within a class
    compounds = set()
    for eles in word_list:
        attribute_dict = extract_attributes(nlp, [eles])
        attributes = list(attribute_dict.keys())
        attributes_arr.append(attributes)
        for c in attributes:
            attributes_count[c] = attributes_count.get(c, 0) + 1

    for c in attributes_count:
        attributes_count[c] = attributes_count[c] / len(word_list)

    attributes_count = [(k, c) for k, c in attributes_count.items()]
    attributes_count = sorted(
        attributes_count, key=lambda x: -x[1]
    )  # first attribute is the most common
    sorted_attributes = [t[0] for t in attributes_count]
    all_attributes2idx = {k: i for i, (k, c) in enumerate(attributes_count)}
    img_embeds = []
    for attributes in attributes_arr:
        embed = np.zeros(len(attributes_count))
        for c in attributes:
            embed[all_attributes2idx[c]] = 1
        img_embeds.append(embed)
    img_embeds = np.array(img_embeds)
    return img_embeds, sorted_attributes


def main(dataset, split, caption_model="vit-gpt2"):
    data_folder = get_data_folder(dataset)
    caption_path = os.path.join(data_folder, f"{split}_{caption_model}_captions.csv")

    save_path = os.path.split(caption_path)[0]
    attribute_save_path = os.path.join(
        save_path, f"{split}_{caption_model}_attribute_embeds.pickle"
    )
    if os.path.exists(attribute_save_path):
        print(f"{attribute_save_path} exists")
        return

    # nlp = spacy.load("en_core_web_trf")
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)

    caption_dict = {}
    with open(caption_path, "r") as f:
        for i, line in enumerate(f):
            eles = line.split(",")
            file_name = eles[0].strip()
            label = eles[-1].strip()
            caption = ", ".join(eles[1:-1])

            if label in caption_dict:
                caption_dict[label].append([file_name, i, caption])
            else:
                caption_dict[label] = [[file_name, i, caption]]

    # within label
    all_class_embeds = {}
    for label in tqdm(caption_dict):
        nums = len(caption_dict[label])
        file_names = [t[0] for t in caption_dict[label]]
        file_idxs = [t[1] for t in caption_dict[label]]
        captions = [t[2] for t in caption_dict[label]]
        img_embeds, attributes = get_img_embed(nlp, captions)

        all_class_embeds[label] = (img_embeds, attributes, file_names, file_idxs)

    with open(attribute_save_path, "wb") as outfile:
        pickle.dump(all_class_embeds, outfile)


if __name__ == "__main__":
    print("Process miniImageNet")
    main("miniImageNet")
    print("Process tieredImageNet")
    main("tieredImageNet")
