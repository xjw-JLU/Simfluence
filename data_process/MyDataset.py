# coding: utf-8
from torch.utils.data import Dataset
import json


class MyDataset(Dataset):
    def __init__(self, nums):
        items = []
        for i in range(nums):
            file_path = f"/root/paddlejob/workspace/env_run/xiaojingwu/Simfluence/dataset/training_run/training_run{i}.json"
            with open(file_path, 'r') as f:
                tot_step = []
                eval_loss = []
                for line in f:
                    step = json.loads(line)
                    tot_step.append(step['train_data_step'])
                    eval_loss.append(step['eval_loss'])
            items.append((tot_step, eval_loss))
        self.items = items

    def __getitem__(self, index):
        # return ((1,2), 3)
        return {"train_data":self.items[index][0], "train_loss":self.items[index][1]}

    def __len__(self):
        return len(self.items)

class MyDataset2(Dataset):
    def __init__(self, file_path, num_samples_to_sample=-1):
        items = []
        cnt = 0

        with open(file_path, 'r') as f:
            for line in f:
                if cnt < num_samples_to_sample:
                    item = json.loads(line)
                    items.append((item['train_data'], item['train_loss']))
                cnt += 1
        self.items = items

    def __getitem__(self, index):
        # return ((1,2), 3)
        return {"train_data":self.items[index][0], "train_loss":self.items[index][1], "id":index}

    def __len__(self):
        return len(self.items)