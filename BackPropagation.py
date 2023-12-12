import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义激活函数及其导数 (使用sigmoid/Relu/LeakRelu/线性函数)
def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)  # 限制 x 的范围以防止 exp 溢出
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def linear_func(x):
    return x


def linear_func_derivative(x):
    return 1


# 初始化网络参数
input_size = 2              # 输入层节点数
hidden_size_1 = 30          # 隐藏层1节点数
hidden_size_2 = 30          # 隐藏层2节点数
hidden_size_3 = 30          # 隐藏层3节点数
output_size = 1             # 输出层节点数

learning_rate = 0.000005    # 学习率
epochs = 5000              # 迭代次数

# 定义输入和输出
x1_values = np.linspace(-0.5, 0.5, 100)             # 训练集区间
x2_values = np.linspace(-0.5, 0.5, 100)
X1, X2 = np.meshgrid(x1_values, x2_values)
# 目标函数
y_values = 5 * (np.sin(X1 + X2) * np.exp(-np.abs(X1 - X2)))
input_data = np.column_stack((X1.ravel(), X2.ravel()))
output_data = y_values.ravel().reshape(-1, 1)

# 随机初始化权重
# np.random.seed(0)
# weights_input_hidden1 = np.random.uniform(size=(input_size, hidden_size_1))
# weights_hidden1_hidden2 = np.random.uniform(size=(hidden_size_1, hidden_size_2))
# weights_hidden2_output = np.random.uniform(size=(hidden_size_2, output_size))
np.random.seed(0)
std_dev1 = np.sqrt(2. / input_size)
weights_input_hidden1 = np.random.normal(0, std_dev1, (input_size, hidden_size_1))
std_dev2 = np.sqrt(2. / hidden_size_1)
weights_hidden1_hidden2 = np.random.normal(0, std_dev2, (hidden_size_1, hidden_size_2))
std_dev3 = np.sqrt(2. / hidden_size_2)
weights_hidden2_hidden3 = np.random.normal(0, std_dev3, (hidden_size_2, hidden_size_3))
std_dev4 = np.sqrt(2. / hidden_size_3)
weights_hidden3_output = np.random.normal(0, std_dev3, (hidden_size_3, output_size))

# 定义损失存储列表
loss_history = []

# 初始化参数
momentum = 0.95  # 动量因子

# 初始化权重的动量项为0
v_weights_input_hidden1 = np.zeros_like(weights_input_hidden1)
v_weights_hidden1_hidden2 = np.zeros_like(weights_hidden1_hidden2)
v_weights_hidden2_hidden3 = np.zeros_like(weights_hidden2_hidden3)
v_weights_hidden3_output = np.zeros_like(weights_hidden3_output)

# 定义上一次的 loss
last_loss = 5

# 训练
for epoch in range(epochs):

    # 前向传播
    hidden_layer1_input = np.dot(input_data, weights_input_hidden1)
    hidden_layer1_output = leaky_relu(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = leaky_relu(hidden_layer2_input)

    hidden_layer3_input = np.dot(hidden_layer2_output, weights_hidden2_hidden3)
    hidden_layer3_output = sigmoid(hidden_layer3_input)

    output_layer_input = np.dot(hidden_layer3_output, weights_hidden3_output)
    output_layer_output = linear_func(output_layer_input)

    # 计算损失,计算目标函数
    loss = np.mean(np.square(output_data - output_layer_output))
    loss_history.append(loss)

    # 设置自适应学习率
    if loss < last_loss:
        learning_rate *= 1.02
    else:
        learning_rate *= 0.95
    last_loss = loss

    # 反向传播
    output_layer_error = output_data - output_layer_output
    output_layer_delta = output_layer_error * linear_func_derivative(output_layer_output)

    hidden_layer3_error = np.dot(output_layer_delta, weights_hidden3_output.T)
    hidden_layer3_delta = hidden_layer3_error * sigmoid_derivative(hidden_layer3_output)

    hidden_layer2_error = np.dot(hidden_layer3_delta, weights_hidden2_hidden3.T)
    hidden_layer2_delta = hidden_layer2_error * leaky_relu_derivative(hidden_layer2_output)

    hidden_layer1_error = np.dot(hidden_layer2_delta, weights_hidden1_hidden2.T)
    hidden_layer1_delta = hidden_layer1_error * leaky_relu_derivative(hidden_layer1_output)

    # # 更新权重
    # weights_hidden3_output += learning_rate * np.dot(hidden_layer3_output.T, output_layer_delta)
    # weights_hidden2_hidden3 += learning_rate * np.dot(hidden_layer2_output.T, hidden_layer3_delta)
    # weights_hidden1_hidden2 += learning_rate * np.dot(hidden_layer1_output.T, hidden_layer2_delta)
    # weights_input_hidden1 += learning_rate * np.dot(input_data.T, hidden_layer1_delta)

    # 更新权重，并加入动量项
    v_weights_input_hidden1 = momentum * v_weights_input_hidden1 + learning_rate * np.dot(input_data.T,
                                                                                          hidden_layer1_delta)
    weights_input_hidden1 += v_weights_input_hidden1

    v_weights_hidden1_hidden2 = momentum * v_weights_hidden1_hidden2 + learning_rate * np.dot(hidden_layer1_output.T,
                                                                                              hidden_layer2_delta)
    weights_hidden1_hidden2 += v_weights_hidden1_hidden2

    v_weights_hidden2_hidden3 = momentum * v_weights_hidden2_hidden3 + learning_rate * np.dot(hidden_layer2_output.T,
                                                                                              hidden_layer3_delta)
    weights_hidden2_hidden3 += v_weights_hidden2_hidden3

    v_weights_hidden3_output = momentum * v_weights_hidden3_output + learning_rate * np.dot(hidden_layer3_output.T,
                                                                                            output_layer_delta)
    weights_hidden3_output += v_weights_hidden3_output

    # 打印损失
    if epoch % 1000 == 0:
        print('Epoch: %d, Loss: %.5f' % (epoch, loss))

# 损失可视化(Loss)
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 在子图上绘制放大的部分
left, bottom, width, height = [0.4, 0.3, 0.3, 0.3]
ax2 = plt.axes([left, bottom, width, height])
ax2.plot(loss_history[:200], color='tab:orange')
plt.show()


# 训练集对比（使用三维图显示）
# 使用神经网络生成预测结果
hidden_layer1_input = np.dot(input_data, weights_input_hidden1)
hidden_layer1_output = leaky_relu(hidden_layer1_input)
hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
hidden_layer2_output = leaky_relu(hidden_layer2_input)
hidden_layer3_input = np.dot(hidden_layer2_output, weights_hidden2_hidden3)
hidden_layer3_output = sigmoid(hidden_layer3_input)
output_layer_input = np.dot(hidden_layer3_output, weights_hidden3_output)
output_layer_output = output_layer_input

# 计算测试集上的性能指标
mse = mean_squared_error(output_data, output_layer_output)
rmse = np.sqrt(mse)
mae = mean_absolute_error(output_data, output_layer_output)
r2 = r2_score(output_data, output_layer_output)
# 计算误差
errors = output_data - output_layer_output

# 计算每个点的期望值的5%
thresholds = 0.05 * np.abs(output_data)

# 比较误差是否小于对应期望值的5%
error_comparison = np.abs(errors) > thresholds

# 找出最大的符合条件的误差
max_compliant_error = np.max(np.abs(errors[error_comparison]))
# 找出最大绝对误差及其索引
max_absolute_error = np.max(np.abs(errors))
max_error_index = np.argmax(np.abs(errors))
# 获取该误差对应的期望值
corresponding_expected_value = output_data[max_error_index]
# 计算比值
error_ratio = max_absolute_error / np.abs(corresponding_expected_value)

# 打印结果
print(f"最大绝对误差: {max_absolute_error}")
print(f"对应的期望值: {corresponding_expected_value}")
print(f"误差比值: {error_ratio}")
# 打印性能指标
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R² Score: {r2}')

# 绘制原始函数的三维图
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X1, X2, y_values.reshape(X1.shape), cmap='viridis')
ax1.set_title('Original Function')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')

# 绘制神经网络拟合的结果的三维图
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X1, X2, output_layer_output.reshape(X1.shape), cmap='viridis')
ax2.set_title('NN Fitted Function')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
plt.show()

# 定义测试区间，测试网络泛化性能
x1_values_test = np.linspace(-0.6, 0.6, 100)
x2_values_test = np.linspace(-0.6, 0.6, 100)
X1_test, X2_test = np.meshgrid(x1_values_test, x2_values_test)
y_values_test = 5 * (np.sin(X1_test + X2_test) * np.exp(-np.abs(X1_test - X2_test)))    # 目标拟合函数
input_data_test = np.column_stack((X1_test.ravel(), X2_test.ravel()))
output_data_test = y_values_test.ravel().reshape(-1, 1)

# 使用神经网络生成预测结果
hidden_layer1_input = np.dot(input_data_test, weights_input_hidden1)
hidden_layer1_output = leaky_relu(hidden_layer1_input)
hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
hidden_layer2_output = leaky_relu(hidden_layer2_input)
hidden_layer3_input = np.dot(hidden_layer2_output, weights_hidden2_hidden3)
hidden_layer3_output = sigmoid(hidden_layer3_input)
output_layer_input = np.dot(hidden_layer3_output, weights_hidden3_output)
output_layer_output = output_layer_input

# 计算测试集上的性能指标
mse = mean_squared_error(output_data_test, output_layer_output)
rmse = np.sqrt(mse)
mae = mean_absolute_error(output_data_test, output_layer_output)
r2 = r2_score(output_data_test, output_layer_output)
# 计算误差
errors = output_data_test - output_layer_output

# 计算每个点的期望值的5%
thresholds = 0.05 * np.abs(output_data_test)

# 比较误差是否小于对应期望值的5%
error_comparison = np.abs(errors) > thresholds

# 找出最大的符合条件的误差
max_compliant_error = np.max(np.abs(errors[error_comparison]))
# 找出最大绝对误差及其索引
max_absolute_error = np.max(np.abs(errors))
max_error_index = np.argmax(np.abs(errors))
# 获取该误差对应的期望值
corresponding_expected_value = output_data_test[max_error_index]
# 计算比值
error_ratio = max_absolute_error / np.abs(corresponding_expected_value)

# 打印结果
print(f"最大绝对误差: {max_absolute_error}")
print(f"对应的期望值: {corresponding_expected_value}")
print(f"误差比值: {error_ratio}")
# 打印性能指标
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R² Score: {r2}')

# 绘制原始函数的等高图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(X1_test, X2_test, y_values_test.reshape(X1_test.shape))
plt.colorbar()
plt.title('Original Function')
plt.xlabel('x1')
plt.ylabel('x2')

# 绘制神经网络拟合的结果的等高线图
plt.subplot(1, 2, 2)
plt.contourf(X1_test, X2_test, output_layer_output.reshape(X1_test.shape))
plt.colorbar()
plt.title('NN Fitted Function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
