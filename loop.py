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



lr_init = 0.1

max_epoch = 4
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

train_path = "/root/paddlejob/workspace/env_run/xiaojingwu/Simfluence/dataset/json_file/train_data_epoch0.json"
valid_path = "/root/paddlejob/workspace/env_run/xiaojingwu/Simfluence/dataset/json_file/train_data_epoch1.json"


train_data = MyDataset(train_path, num_samples_to_train)
# valid_data = MyDataset(train_path, num_samples_to_valid)
valid_data = train_data
print("train_data: ", len(train_data))
print("train_data: ", len(valid_data))




num_train_samples = len(train_data)
# valid_data = MyDataset(valid_path)
# valid_num = len(valid_data)

# valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)



net = MetricSimulator(num_train_samples)    
# net.initialize_weights()    


criterion = nn.MSELoss()                                                  
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)    


device = "cuda:0" if torch.cuda.is_available() else "cpu"

M_prev_prev = train_data[0]["train_loss"]
M_prev = train_data[0]["train_loss"]

train_losses = []
vaild_losses = []
train_epochs_losses = []
valid_epochs_losses = []
net.train()
for run_num in range(train_run):

    selected_subset, _ = random_split(train_data, [num_samples_to_select, len(train_data) - num_samples_to_select])
    train_loader = DataLoader(dataset=selected_subset, batch_size=train_bs, shuffle=True)

    for epoch in range(max_epoch):

        # loss_sigma = 0.0   
        # correct = 0.0
        # total = 0.0
        epoch_iterator = tqdm(train_loader, desc="Training")
        
        total_epoch = max_epoch * run_num + epoch
        for step, data in enumerate(epoch_iterator):
            
            inputs, labels, ids = data["train_data"], data["train_loss"], data["id"]
            
            ids = torch.tensor(ids)
            labels = torch.tensor(labels)

            outputs = net(ids, M_prev, labels)
            # print("output:", outputs.dtype, outputs)
            # print("labels:", labels.dtype, labels.float())
            loss = criterion(outputs, labels)
            # print("loss",loss)
            # loss.requires_grad = True
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss",loss)
            # loss_sigma += loss.item()
            
            train_losses.append(loss.item())   
            

            
            max_tot_epoch = max_epoch * train_run
            print("Training: step[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                total_epoch + 1, max_tot_epoch, step + 1, len(train_loader), loss))

            train_step = len(train_loader)*total_epoch+step+1

            # writer.add_scalars('Loss_group', {'loss': loss}, train_step)
            # writer.add_scalar('train_loss', loss, train_step)
            writer.add_scalars('train_loss', {"train_loss":loss}, train_step)
        writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            

        with open(f"/root/paddlejob/workspace/env_run/xiaojingwu/Simfluence/output/train_losses_epoch{total_epoch}.json", "a") as f:
            data = json.dumps(train_losses)
            f.write(data+'\n')
        train_losses = []

        # scheduler.step()
        
        # 每个epoch，记录梯度，权值
        # for name, layer in net.named_parameters():
        #     writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        #     writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)


net.eval()
for run_num in range(val_run):
    selected_subset, _ = random_split(valid_data, [num_samples_to_select, len(valid_data) - num_samples_to_select])
    valid_loader = DataLoader(dataset=selected_subset, batch_size=train_bs, shuffle=True)
    
    for epoch in range(max_epoch):
        # loss_sigma = 0.0    # 记录一个epoch的loss之和
        # correct = 0.0
        # total = 0.0
        epoch_iterator = tqdm(valid_loader, desc="Testing")
          
        for step, data in enumerate(epoch_iterator):
            
            inputs, labels, ids = data["train_data"], data["train_loss"], data["id"]
            
            ids = torch.tensor(ids)
            labels = torch.tensor(labels)

            outputs = net(ids, M_prev, labels)
            loss = criterion(outputs, labels)
            # print("outputs:", outputs)
            # print("lagels:", label)
            
            total_epoch = max_epoch * run_num + epoch
            max_tot_epoch = max_epoch * train_run
            print("Testing: step[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                total_epoch + 1, max_tot_epoch, step + 1, len(valid_loader), loss))

            valid_step = len(valid_loader)*total_epoch+step+1

            # writer.add_scalars('Loss_group', {'vaild_loss': loss}, train_step)
            writer.add_scalars('vaild_loss', {"test_loss":loss}, valid_step)
            
            writer.add_scalars('predit_result', {'true_loss': labels.mean().item(),"predit_loss": outputs.mean().item()}, valid_step)
            




    
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