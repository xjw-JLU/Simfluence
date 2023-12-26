# coding: utf-8

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tensorboardX import SummaryWriter
from datetime import datetime
from data_process.MyDataset import MyDataset
from model.MetricSimulator import MetricSimulator
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from torch.utils.data.dataset import random_split




max_epoch = 15
train_run = 22
val_run = 10
train_bs = 16
valid_bs = 16
num_samples_to_train = 100
num_samples_to_valid = 100
num_samples_to_select = 64


log_dir = "log2"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)


all_nums = 32
dataset = MyDataset(all_nums)

valid_num = 10
train_num = all_nums - valid_num

train_data, valid_data = random_split(dataset, [train_num, valid_num])

print("train_data: ", len(train_data))
print("valid_num: ", len(valid_data))

num_train_samples = len(train_data)
# valid_data = MyDataset(valid_path)
# valid_num = len(valid_data)

# valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

data_to_index ={}
mset = set()
cnt = 0
for data in dataset:
    tot_step = data["train_data"]
    for step in tot_step:
        for data in step:
            if data not in mset:
                mset.add(data)
                data_to_index[data] = cnt
                cnt += 1
print("len_mset:", len(mset))
num_train_samples = len(mset)
net = MetricSimulator(num_train_samples, data_to_index)
net.initialize_weights()    


criterion = nn.MSELoss(reduction="mean")                                                
optimizer = optim.Adam(net.parameters(), lr=0.0009, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.01)    


device = "cuda:0" if torch.cuda.is_available() else "cpu"

M_prev_prev = train_data[0]["train_loss"]
M_prev = train_data[0]["train_loss"]

train_losses = []

net.train()
for epoch in range(max_epoch):
    scheduler.step()  # 更新学习率
    for step, data in enumerate(train_data):
        # scheduler.step()  # 更新学习率
        inputs, labels = data["train_data"], data["train_loss"]
        labels = torch.tensor(labels)
        M_prev = labels[0]
        outputs = net(inputs, M_prev)
        # print("output:", outputs.dtype, outputs)
        # print("labels:", labels.dtype, labels.float())
        loss = criterion(outputs, labels)
        # print("loss",loss)
        # loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss_sigma += loss.item()
        train_losses.append(loss.item())   
        
        print("Training: step[{:0>3}/{:0>3}] Loss: {:.4f} ".format(step + 1, len(train_data), loss))

        writer.add_scalars('train_loss', {"train_loss":loss}, step)
    
    writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
        
with open(f"/root/paddlejob/workspace/env_run/xiaojingwu/Simfluence/output/train_losses.json", "w") as f:
    data = json.dumps(train_losses)
    f.write(data+'\n')
    # train_losses = []


net.eval()
# for run_num in range(val_run):
#     selected_subset, _ = random_split(valid_data, [num_samples_to_select, len(valid_data) - num_samples_to_select])
#     valid_loader = DataLoader(dataset=selected_subset, batch_size=train_bs, shuffle=True)
    
#     for epoch in range(max_epoch):
#         # loss_sigma = 0.0    # 记录一个epoch的loss之和
#         # correct = 0.0
#         # total = 0.0
#         epoch_iterator = tqdm(valid_loader, desc="Testing")
          
#         for step, data in enumerate(epoch_iterator):
            
#             inputs, labels, ids = data["train_data"], data["train_loss"], data["id"]
            
#             ids = torch.tensor(ids)
#             labels = torch.tensor(labels)

#             outputs = net(ids, M_prev, labels)
#             loss = criterion(outputs, labels)
#             # print("outputs:", outputs)
#             # print("lagels:", label)
            
#             total_epoch = max_epoch * run_num + epoch
#             max_tot_epoch = max_epoch * train_run
#             print("Testing: step[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
#                 total_epoch + 1, max_tot_epoch, step + 1, len(valid_loader), loss))

#             valid_step = len(valid_loader)*total_epoch+step+1

#             # writer.add_scalars('Loss_group', {'vaild_loss': loss}, train_step)
#             writer.add_scalars('vaild_loss', {"test_loss":loss}, valid_step)
            
#             writer.add_scalars('predit_result', {'true_loss': labels.mean().item(),"predit_loss": outputs.mean().item()}, valid_step)
            
vaild_losses = []

#每个元素都是一个列表元素，该列表元素表示测试集中的一个run的loss序列
predit_losses = []
true_losses = []
tot_losses = []
for step, data in enumerate(valid_data):
    
    inputs, labels = data["train_data"], data["train_loss"]
    labels = torch.tensor(labels)
    M_prev = labels[0]
    outputs = net(inputs, M_prev)
    
    # print("output:", outputs.dtype, outputs)
    # print("labels:", labels.dtype, labels.float())
    loss = criterion(outputs, labels)
    # predit_losses.append(outputs.tolist())
    # true_losses.append(labels.tolist())
    tot_losses.append((labels.tolist(), outputs.tolist()))
    # loss_sigma += loss.item()
    
    print("Testing: step[{:0>3}/{:0>3}] Loss: {:.4f} ".format(step + 1, len(train_data), loss))

    writer.add_scalars('vaild_loss', {"test_loss":loss}, step)
    
with open("tot_losses.json",'w') as f:
    f.write(json.dumps(tot_losses))

  
    
# ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
net_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(net.state_dict(), net_save_path)

# train_acc = validate(net, train_loader, 'train', classes_name)
# valid_acc = validate(net, valid_loader, 'valid', classes_name)

# plt.figure(figsize=(12,4))
# plt.subplot(121)
# plt.plot(train_loss[:])
# plt.title("train_loss")
# plt.subplot(122)
# plt.plot(train_epochs_loss[1:],'-o',label="train_loss")
# # plt.plot(valid_epochs_loss[1:],'-o',label="valid_loss")
# plt.title("epochs_loss")
# plt.legend()
# plt.show()