### loop2.py
模拟器执行入口

### data_process/MyDataset.py
每一个example对应gpt2finetune的一个run
一个run是一个序列 [step_1,step_2,..,step_n] [loss_1,loss_2,...,loss_n]

### MetricSimulator.py
模拟器模型

(1) 参数: 一个$n*2$的矩阵, n对应finetune训练样本的总数

(2) 输入: $curriculum=[c_1,c_2,...,c_n]$ ，ci表示的是在当前run中的第i个step微调用的训练样本, 对应上面的例子就是4个训练样本

(3) 输出: 预测loss序列$[\hat{L_1} ( Z_{test}),\hat{L_2} (Z_{test}),..., \hat{L_n}(Z_{test})]$  


### test.ipynb
实验结果打印
