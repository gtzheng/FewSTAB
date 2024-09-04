# FewSTAB: Few-Shot Tasks with Attribute Biases

## Preparations
### Dataset
Please follow the instructions in [LibFewShot](https://github.com/RL-VIG/LibFewShot) to prepare datasets.
### Prepare few-shot classifiers
Change the config file path at line 19 in `run_trainer.py`. Then train a few-shot classifier using the following command:
```bash
python run_trainer.py
```
### Extract attributes
Use the following command to extract attributes from a dataset using a specified VLM (vit-gpt2 or blip):
```bash
python extract_attributes.py --dataset miniImagenet --model vit-gpt2
```
Upon completion, you will get `test_vit-gpt2_captions.csv` and `test_vit-gpt2_attribute_embeds.pickle`.

### Configure `spurious/config.py`
Specify the data root path on line 3, paths to few-shot classifiers from line 34 to line 143, and extracted attributes from line 146 to line 209.

## Usage 
### Within the framework
To test a model's robustness to spurious bias:
```bash
python run_test.py --method leo --spurious --test_shot 1 --train_shot 1 --test_episode 3000 --dataset miniImagenet
```

To do normal few-shot evaluation
```bash
python run_test.py --method leo --test_shot 1 --train_shot 1 --test_episode 3000 --dataset miniImagenet
```
`train_shot` means we test model trained with `train_shot` tasks.
We can test a model's robustness to spurious bias under different shot numbers in a task by fixing `train_shot` and changing `test_shot`.

### Outside the framework
First, generate evaluation tasks using the following code.
```python
from spurious.spurious_task import SpuriousTask, SpuriousSamplerSimple

test_task_path = "spurious_test_3000E_5w1s.pt"
attribute_path_test = "test_vit-gpt2_attribute_embeds.pickle"
# construct a SpuriousTask object (no task generation in this step)
attr_task_obj = SpuriousTask(
                attribute_path_test, # path to the extracted attributes
                3000, # number of test tasks
                5, # number of classes per task
                1, # number of suppoer samples per class
                15, # number of query samples per class
                seed=0, # random seed; we use 0 in our paper
            )

# bulid a test sampler in the offline mode (will generate tasks)
test_sampler = SpuriousSamplerSimple(
        attr_task_obj,
        test_task_path, # 
        3000, # number of tasks sampled (same as those in attr_task_obj)
        5, # number of classes per task
        1, # number of support samples per class
        15, # number of query samples per class
        1, # number of tasks per batch
        mode="offline",
    )
```
Then, create a test data loader for testing.
```python
from core.data.miniImagenet import miniImagenet
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
# Create a dataset object. Here, we use miniImagenet in the repository as an example. You can create your own.
transforms = Compose([
    CenterCrop(size=(224, 224)),
    ToTensor(),
    Normalize(mean=torch.tensor([0.4815, 0.4578, 0.4082]), std=torch.tensor([0.2686, 0.2613, 0.2758]))]
  )

test_dataset = miniImagenet(
        data_root, # path to a dataset folder
        mode="test",
        transform=transforms
    )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler, pin_memory=True, num_workers=8)
tasks, labels = next(iter(test_loader)) # B x 3 x H x W
img_shape = tasks.shape[-3:]
task = tasks.view(1, 5, 1+15, *img_shape)[0]
support = task[:,0:1]
query = task[:,1:]
print("support shape", support.shape) # support shape torch.Size([5, 1, 3, 224, 224])
query = task[:,1:]
print("query shape", query.shape) # shape torch.Size([5, 15, 3, 224, 224])
```

## Citation

Please consider citing this paper if you find the code helpful.
```
@inproceedings{zhengeccv24few,
  title={Benchmarking Spurious Bias in Few-Shot Image Classifiers},
  author={Zheng, Guangtao and Ye, Wenqian and Zhang, Aidong},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Reference
- [LibFewShot](https://arxiv.org/abs/2109.04898)
