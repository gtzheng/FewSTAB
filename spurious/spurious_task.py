import torch
import numpy as np
from collections import Counter
import os
from tqdm import tqdm
import pickle
import time



class SpuriousTask:
    def __init__(
        self,
        eval_attribute_path,
        n_batch,
        n_cls,
        n_s,
        n_q,
        seed=0,
    ):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_s = n_s
        self.n_q = n_q
        with open(eval_attribute_path, "rb") as f:
            eval_attribute_dict = pickle.load(f)
       
       
        (
            self.test_attr_info,
            self.test_attr2classes,
        ) = self.analyze_attributes(eval_attribute_dict)
        num_attrs = len(self.test_attr_info[0])
        self.attr_info = (self.test_attr_info[0], np.ones(num_attrs)/num_attrs)
        self.attributes_dict = eval_attribute_dict
        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        return self.n_batch

    def analyze_attributes(self, test_attrs):
        test_attr_counter = Counter()
        test_attr2classes = {}
        for c in test_attrs:  # embeds, attributes, file_names, file_ids
            cls_attributes = np.array(test_attrs[c][1])
            cls_attributes_sum = test_attrs[c][0].sum(axis=0)
            num_samples = len(test_attrs[c][0])
            test_attr_counter.update(
                Counter(
                    {a: cls_attributes_sum[i] for i, a in enumerate(cls_attributes)}
                )
            )
            sel_attributes = cls_attributes[(cls_attributes_sum >= self.n_s)*(num_samples-cls_attributes_sum >= self.n_q)]
            for a in sel_attributes:
                if a in test_attr2classes:
                    test_attr2classes[a].append(c)
                else:
                    test_attr2classes[a] = [c]

        test_attributes = list(set(test_attr2classes.keys()))
        counts = np.array([test_attr_counter[a] for a in test_attributes])
        probs = counts / counts.sum()
        test_attrs_info = (test_attributes, probs)
        return test_attrs_info, test_attr2classes

    def generate(self, save_path=None):
        if save_path is not None:
            if os.path.exists(save_path):
                print(f"{save_path} exists, skipping")
                return
        
        total = 0
        batch = []
        for i_batch in tqdm(
            range(self.n_batch), leave=False, desc="Spurious-Few-Shot-Task-Genenration"
        ):
            failed_attempts_support = 0
            failed_attempts_query = 0
            query_threshold = 0
            while True:
                episode = []
                episode_attr = []
                cls_attr_sets = []
                success = True

                sel_attrs = self.rng.choice(
                    self.attr_info[0],
                    self.n_cls,
                    p=self.attr_info[1],
                    replace=False,
                )


                sel_classes = []
                for i_a in range(self.n_cls):
                    cls_attr = set(self.test_attr2classes[sel_attrs[i_a]])
                    cls_attr_remains = [
                        self.test_attr2classes[sel_attrs[i_b]]
                        for i_b in range(self.n_cls)
                        if i_b != i_a
                    ]
                    cls_attr_remains = set(
                        [c for classes in cls_attr_remains for c in classes]
                    )
                    overlapped_classes = cls_attr.intersection(cls_attr_remains)
                    if len(overlapped_classes) == 0:
                        success = False
                        break
                    remain_classes = list(overlapped_classes-set(sel_classes)) #make sure there are no duplicate classes
                    if len(remain_classes) == 0:
                        success = False
                        break
                    cls = self.rng.choice(remain_classes)
                    sel_classes.append(cls)
                    img_embed = self.attributes_dict[cls][0]
                    cls_all_attrs = self.attributes_dict[cls][1]
                    cls_attr2idx = {a: i for i, a in enumerate(cls_all_attrs)}
 
                    all_attr_indexes = np.arange(img_embed.shape[0])

                    attr_idx_w = cls_attr2idx[sel_attrs[i_a]]
                    attr_idx_wo = np.array(
                        [
                            cls_attr2idx[sel_attrs[i_b]]
                            for i_b in range(self.n_cls)
                            if i_b != i_a and sel_attrs[i_b] in cls_attr2idx
                        ]
                    )
                    if len(attr_idx_wo) == 0:
                        success = False
                        break
                    # for selecting support set samples
                    sel_criterion = (img_embed[:, attr_idx_w] == 1) * (
                        img_embed[:, attr_idx_wo].sum(axis=1) <= failed_attempts_support
                    )
                    sel_indexes = all_attr_indexes[sel_criterion]
                    if len(sel_indexes) < self.n_s:
                        failed_attempts_support += 0.2
                        success = False
                        break
                    failed_attempts_support = 0
                    l_s = self.rng.choice(sel_indexes, self.n_s, replace=False)

                    # for selecting query set samples
                    sel_criterion = (img_embed[:, attr_idx_w] == 0) * (
                        img_embed[:, attr_idx_wo].mean(axis=1) > query_threshold
                    )
                    sel_indexes = all_attr_indexes[sel_criterion]
                    if len(sel_indexes) < self.n_q:
                        success = False
                        failed_attempts_query += 1
                        if failed_attempts_query == 5:
                            query_threshold = -1
                        break
                    failed_attempts_query = 0
                    query_threshold = 0
                    remain_attrs = set(self.attr_info[0]) - set(sel_attrs)
                    remain_idxs = np.array(
                        [cls_attr2idx[a] for a in remain_attrs if a in cls_attr2idx]
                    )
                    prob_attrs = img_embed[sel_indexes][:, remain_idxs].mean(axis=0)

                    scores = np.log(
                        img_embed[sel_indexes][:, remain_idxs]
                        * prob_attrs.reshape(1, -1)
                        + 1e-10
                    ).sum(axis=1)
                    l_q = sel_indexes[np.argsort(scores)[0 : self.n_q]]

                    episode.append([l for l in np.concatenate((l_s, l_q))])
                    episode_attr.append(sel_attrs)
                if not success:
                    continue
                else:
                    break

            episode_ori = torch.tensor(
                [
                    [self.attributes_dict[c][3][i] for i in episode[i_c]]
                    for i_c, c in enumerate(sel_classes)
                ]
            )
            batch.append((episode_ori, episode_attr))

        if save_path is not None:
            torch.save(batch, save_path)
        return batch


class SpuriousSampler:
    def __init__(
        self,
        task_obj,
        path,
        n_batches,
        n_cls,
        n_s,
        n_q,
        ep_per_batch=1,
        mode="offline",
        worst_group=True,
    ):
        if mode == "offline":
            task_obj.generate(path)
            self.data = torch.load(path)
        else:
            self.data = None
            self.task_obj = task_obj
        self.mode = mode
        self.worst_group = worst_group
        # # sanity check
        # assert (
        #     len(self.data) == n_batches
        # ), f"len(self.data) ({len(self.data)}) != {n_batches}"
        # n_way, num_per_cls = self.data[0][0].shape
        # assert n_way == n_cls, f"n_way ({n_way}) != {n_cls}"
        # assert (
        #     num_per_cls == n_s + n_q
        # ), f"num_per_cls({num_per_cls}) != n_s ({n_s}) + n_q ({n_q})"
        self.n_batches = n_batches
        self.ep_per_batch = ep_per_batch

        self.n_s = n_s
        self.n_q = n_q
        self.n_cls = n_cls

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.mode == "online":
            start_time = time.time()
            data = self.task_obj.generate()
            elapsed = time.time() - start_time
            print(f"average time {elapsed/self.n_batches}")
            for n in range(self.n_batches):
                batch = []
                for _ in range(self.ep_per_batch):
                    batch.append(data[n][0])
                batch = torch.stack(batch).reshape(-1)
                yield batch
        else:
            data = self.data
            if self.worst_group:
                assert (
                    self.ep_per_batch == 1
                ), "episode_size == 1 for calculating the worst-group accuracy"
                batch = []
                for n in range(self.n_batches):
                    for c in range(self.n_cls):
                        cls_batch = torch.cat(
                            (
                                data[n][0][:, 0 : self.n_s],
                                data[n][0][c, self.n_s :].repeat(self.n_cls, 1),
                            ),
                            dim=1,
                        )
                        batch.append(cls_batch)
                    batch = torch.stack(batch).reshape(-1)
                    yield batch
                    batch = []
            else:
                for n in range(self.n_batches):
                    batch = []
                    for _ in range(self.ep_per_batch):
                        batch.append(data[n][0])
                    batch = torch.stack(batch).reshape(-1)
                    yield batch



class SpuriousSamplerSimple:
    def __init__(
        self,
        task_obj,
        path,
        n_batches,
        n_cls,
        n_s,
        n_q,
        ep_per_batch=1,
        mode="offline",
    ):
        if mode == "offline":
            task_obj.generate(path)
            self.data = torch.load(path)
        else:
            self.data = None
            self.task_obj = task_obj
        self.mode = mode

        self.n_batches = n_batches
        self.ep_per_batch = ep_per_batch

        self.n_s = n_s
        self.n_q = n_q
        self.n_cls = n_cls

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.mode == "online":  # for training
            start_time = time.time()
            data = self.task_obj.generate()
            elapsed = time.time() - start_time
            print(f"average time {elapsed/self.n_batches}")
            for n in range(self.n_batches):
                batch = []
                for _ in range(self.ep_per_batch):
                    batch.append(data[n][0])
                batch = torch.stack(batch).reshape(-1)
                yield batch
        else:
            data = self.data
            for n in range(self.n_batches):
                batch = []
                for _ in range(self.ep_per_batch):
                    batch.append(data[n][0])
                batch = torch.stack(batch).reshape(-1)
                yield batch