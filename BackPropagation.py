import numpy as np
import matplotlib.pyplot as plt


# 定义激活函数及其导数 (使用sigmoid/Relu/线性函数)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def linear_func(x):
    return x


def linear_func_derivative(x):
    return 1


# 初始化网络参数
input_size = 1          # 输入层节点数
hidden_size_1 = 8       # 隐藏层节点数
hidden_size_2 = 8
output_size = 1         # 输出层节点数

learning_rate = 0.01     # 学习率
epochs = 50000          # 迭代次数

# 定义输入和输出
x_values = np.arange(-0.5, 0.45, 0.05)          # 训练集区间
y_values = np.sin(10 * x_values) * np.exp(-1.9 * (x_values + 0.5))                 # 目标拟合函数
input_data = x_values.reshape(-1, 1)
output_data = y_values.reshape(-1, 1)

# 随机初始化权重
np.random.seed(0)
weights_input_hidden1 = np.random.uniform(size=(input_size, hidden_size_1))
weights_hidden1_hidden2 = np.random.uniform(size=(hidden_size_1, hidden_size_2))
weights_hidden2_output = np.random.uniform(size=(hidden_size_2, output_size))

# 定义损失存储列表
loss_history = []

# 训练
for epoch in range(epochs):
    # 前向传播
    hidden_layer1_input = np.dot(input_data, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
    output_layer_output = linear_func(output_layer_input)

    # 计算损失,计算目标函数
    loss = np.square(output_data - output_layer_output).sum() / 2
    loss_history.append(loss)

    # 反向传播
    output_layer_error = output_data - output_layer_output
    output_layer_delta = output_layer_error * linear_func_derivative(output_layer_output)

    hidden_layer2_error = np.dot(output_layer_delta, weights_hidden2_output.T)
    hidden_layer2_delta = hidden_layer2_error * sigmoid_derivative(hidden_layer2_output)

    hidden_layer1_error = np.dot(hidden_layer2_delta, weights_hidden1_hidden2.T)
    hidden_layer1_delta = hidden_layer1_error * sigmoid_derivative(hidden_layer1_output)

    # 更新权重
    weights_hidden2_output += learning_rate * np.dot(hidden_layer2_output.T, output_layer_delta)
    weights_hidden1_hidden2 += learning_rate * np.dot(hidden_layer1_output.T, hidden_layer2_delta)
    weights_input_hidden1 += learning_rate * np.dot(input_data.T, hidden_layer1_delta)

    # 打印损失
    if epoch % 10000 == 0:
        print('Epoch: %d, Loss: %.4f' % (epoch, loss))

# 损失可视化
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# 绘制拟合曲线与原曲线
plt.figure(figsize=(10, 6))

# 绘制原始函数
plt.plot(x_values, y_values, label='Original Function')

# 使用神经网络生成预测结果
hidden_layer1_input = np.dot(input_data, weights_input_hidden1)
hidden_layer1_output = sigmoid(hidden_layer1_input)
hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
hidden_layer2_output = sigmoid(hidden_layer2_input)
output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
output_layer_output = output_layer_input

# 绘制神经网络拟合的结果
plt.plot(x_values, output_layer_output, label='NN Fitted Function', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Original Function and NN Fitted Function')
plt.legend()
plt.show()
