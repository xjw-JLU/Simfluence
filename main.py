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


# 创建样本ID到参数索引的映射
sample_ids = {id for batch_ids, _ in train_data for id in batch_ids}
id_to_param_index = {sample_id: i for i, sample_id in enumerate(sample_ids)}
num_samples = len(sample_ids)

# 为每个样本设置参数
A = nn.Parameter(torch.randn(num_samples))
B = nn.Parameter(torch.randn(num_samples))
C = nn.Parameter(torch.randn(num_samples))

# 构建模型
class MetricSimulator(nn.Module):
    def __init__(self, num_samples):
        super(MetricSimulator, self).__init__()
        self.A = A
        self.B = B
        self.C = C

    def forward(self, c_t_indices, M_prev):
        alpha = self.A[c_t_indices].sum()
        beta = self.B[c_t_indices].sum()
        gamma = self.C[c_t_indices].sum()
        M_pred = alpha * M_prev + gamma * M_prev + beta
        return M_pred

# 实例化模拟器
simulator = MetricSimulator(num_samples)

# 损失函数
def loss_fn(real_loss, predicted_loss):
    mse_loss = nn.MSELoss()
    return mse_loss(real_loss, predicted_loss)

# 优化器
optimizer = optim.SGD(simulator.parameters(), lr=0.01)

# 训练模型
losses = []
M_prev = torch.tensor(0.0)  # 初始性能指标（loss）
for batch_ids, real_loss in train_data:
    c_t_indices = torch.tensor([id_to_param_index[sample_id] for sample_id in batch_ids])
    real_loss = torch.tensor(real_loss)
    
    predicted_loss = simulator(c_t_indices, M_prev)
    loss = loss_fn(real_loss, predicted_loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    M_prev = predicted_loss.detach()  # 更新前一个性能指标（loss）
    losses.append(loss.item())

# 绘制损失图
plt.plot(losses)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()