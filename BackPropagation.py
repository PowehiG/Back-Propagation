import numpy as np
from tqdm import tqdm
# 定义激活函数及其导数(使用sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 初始化网络参数
input_size = 2          # 输入层节点数
hidden_size = 3         # 隐藏层节点数
output_size = 1         # 输出层节点数

learning_rate = 0.1    # 学习率
epochs = 10000           # 迭代次数

# 定义输入和输出
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[0], [1], [1], [0]])

# 随机初始化权重
np.random.seed(0)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# 训练
for epoch in range(epochs):
#for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
    # 前向传播
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # 计算损失
    loss = np.square(output_data - output_layer_output).sum() / 2

    # 反向传播
    output_layer_error = output_data - output_layer_output
    output_layer_delta = output_layer_error * sigmoid_derivative(output_layer_output)

    hidden_layer_error = np.dot(output_layer_delta, weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # 更新权重
    weights_hidden_output += learning_rate * np.dot(hidden_layer_output.T, output_layer_delta)
    weights_input_hidden += learning_rate * np.dot(input_data.T, hidden_layer_delta)

    # 打印损失
    print('Epoch: %d, Loss: %.4f' % (epoch, loss))

# 测试
print('Predictions:')
hidden_layer_input = np.dot(input_data, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
output_layer_output = sigmoid(output_layer_input)

print(output_layer_output)