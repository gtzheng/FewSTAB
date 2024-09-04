import torch
import numpy as np
from collections import Counter


def analyze_spurious_tasks(path):
    data = torch.load(path)
    counter = Counter()
    for b in range(len(data)):
        indexes = data[b][0].reshape(-1).numpy()
        counter.update(indexes)
    sorted_items = counter.most_common()
    print(f"number of used samples: {len(counter)}")
    print(f"maximum number of used samples: {sorted_items[0][1]}")
    print(f"minimum number of used samples: {sorted_items[-1][1]}")
    numbers = np.array([eles[1] for eles in sorted_items])
    print(f"Average number: {numbers.mean():.2f}, std: {numbers.std():.2f} ")


if __name__ == "__main__":
    path = "/experiments/few_shot_dataset/miniImageNet--ravi/v2_miniImagenet_spurious_test_3000E_5w5s_vit-gpt2.pt"
    analyze_spurious_tasks(path)
