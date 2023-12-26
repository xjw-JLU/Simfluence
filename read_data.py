import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 加载 JSON 文件
with open('test.json', 'r') as file:
    data = json.load(file)

# 解析 JSON 数据，提取训练样本 ID 和对应的 loss
train_data = []
for batch_str, eval_results in data.items():
    batch_ids = json.loads(batch_str)  # 将字符串转化为列表
    for eval_id, metrics in eval_results.items():
        sample_id = int(eval_id)  # 将评估样本的 ID 转化为标量
        train_data.append((batch_ids,sample_id, metrics['loss']))
print(train_data[0])