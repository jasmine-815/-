import matplotlib
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 或 'Agg'，根据你的环境选择

# 设置随机种子
np.random.seed(42)


# --------------------------
# 数据加载与预处理
# --------------------------
def load_dataset(data_dir, img_size=(32, 32)):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            # 解析标签：cat=0, dog=1
            label = 0 if filename.startswith('cat') else 1

            # 加载并预处理图像
            img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img).transpose(2, 0, 1)  # 转换为CHW格式
            img_array = img_array / 255.0 - 0.5  # 归一化到[-0.5, 0.5]

            images.append(img_array)
            labels.append(label)

    return np.array(images), np.array(labels).reshape(-1, 1)


# --------------------------
# 手动实现核心组件
# --------------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.bias = np.zeros((out_channels, 1))

    def forward(self, x):
        self.input = x
        B, C, H, W = x.shape
        K = self.kernels.shape[3]  # 获取kernel_size
        out_h = H - K + 1
        out_w = W - K + 1

        output = np.zeros((B, self.kernels.shape[0], out_h, out_w))
        for b in range(B):
            for oc in range(self.kernels.shape[0]):
                for h in range(out_h):
                    for w in range(out_w):
                        patch = x[b, :, h:h + K, w:w + K]
                        # 修正求和维度
                        output[b, oc, h, w] = np.sum(patch * self.kernels[oc], axis=(0, 1, 2)) + self.bias[oc].item()
        return output

    def backward(self, grad_output, lr=0.001):
        B, OC, OH, OW = grad_output.shape
        C, K = self.kernels.shape[1], self.kernels.shape[2]
        H, W = self.input.shape[2], self.input.shape[3]

        grad_kernels = np.zeros_like(self.kernels)
        grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(self.input)

        # 计算bias梯度
        for oc in range(OC):
            grad_bias[oc] = np.sum(grad_output[:, oc, :, :])

        # 计算kernel梯度
        for oc in range(OC):
            for c in range(C):
                for ki in range(K):
                    for kj in range(K):
                        patch = self.input[:, c, ki:ki + OH, kj:kj + OW]
                        grad_kernels[oc, c, ki, kj] = np.sum(patch * grad_output[:, oc, :, :])

        # 计算输入梯度
        for b in range(B):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        i_start = max(0, h - (K - 1))
                        i_end = min(OH, h + 1)
                        j_start = max(0, w - (K - 1))
                        j_end = min(OW, w + 1)

                        for oc in range(OC):
                            for i in range(i_start, i_end):
                                for j in range(j_start, j_end):
                                    # 确保索引在有效范围内
                                    kernel_h = h - i + (K - 1)
                                    kernel_w = w - j + (K - 1)
                                    if 0 <= kernel_h < K and 0 <= kernel_w < K:
                                        grad_input[b, c, h, w] += (
                                                self.kernels[oc, c, kernel_h, kernel_w] * grad_output[b, oc, i, j]
                                        )

        # 参数更新
        self.kernels -= lr * grad_kernels / B
        self.bias -= lr * grad_bias / B

        return grad_input


class MaxPool2D:
    def forward(self, x):
        self.input = x
        B, C, H, W = x.shape
        self.output = np.zeros((B, C, H // 2, W // 2))
        self.mask = np.zeros_like(x)

        for b in range(B):
            for c in range(C):
                for h in range(0, H, 2):
                    for w in range(0, W, 2):
                        patch = x[b, c, h:h + 2, w:w + 2]
                        max_val = np.max(patch)
                        self.output[b, c, h // 2, w // 2] = max_val
                        self.mask[b, c, h:h + 2, w:w + 2] = (patch == max_val)
        return self.output

    def backward(self, grad_output):
        return grad_output.repeat(2, axis=2).repeat(2, axis=3) * self.mask


class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = np.float32(alpha)  # 显式转换为float32保证类型一致

    def forward(self, x):
        self.mask = (x > 0)  # 保存布尔掩码
        # 正向传播公式：x > 0时输出x，否则输出alpha*x
        return np.where(self.mask, x, self.alpha * x)

    def backward(self, grad):
        # 反向传播公式：梯度乘以导数值（x>0时为1，否则为alpha）
        return grad * np.where(self.mask, 1.0, self.alpha)


class Flatten:
    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.original_shape)


class Linear:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros((1, out_features))

    def forward(self, x):
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output, lr=0.001):
        grad_weights = self.input.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.weights.T

        self.weights -= lr * grad_weights / grad_output.shape[0]
        self.bias -= lr * grad_bias / grad_output.shape[0]
        return grad_input


class SoftmaxCrossEntropy:
    def forward(self, x, y):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.sum(y * np.log(self.probs + 1e-8)) / x.shape[0]
        return loss

    def backward(self, y):
        return (self.probs - y) / y.shape[0]


# --------------------------
# 定义CNN模型
# --------------------------
class ManualCNN:
    def __init__(self):
        # 输入尺寸：3x32x32
        self.conv1 = Conv2D(3, 16, kernel_size=3)  # 输出：16x30x30
        self.relu1 = LeakyReLU()
        self.pool1 = MaxPool2D()                   # 输出：16x15x15
        self.conv2 = Conv2D(16, 32, kernel_size=2)  # 修改kernel_size为2 → 输出：32x14x14
        self.relu2 = LeakyReLU()
        self.pool2 = MaxPool2D()                    # 输出：32x7x7
        self.flatten = Flatten()                    # 输出：32*7*7=1568
        self.fc1 = Linear(1568, 2)                  # 调整全连接层输入
        self.loss_fn = SoftmaxCrossEntropy()
    def forward(self, x, y=None):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)  # 新增第二层卷积
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)

        if y is not None:
            loss = self.loss_fn.forward(x, y)
            return x, loss
        return x

    def backward(self, y):
        grad = self.loss_fn.backward(y)
        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
        return grad



# --------------------------
# 训练流程
# --------------------------
# 加载数据
X_train, y_train = load_dataset("dataset/train", img_size=(32, 32))
X_test, y_test = load_dataset("dataset/test", img_size=(32, 32))

# 打乱训练数据
train_indices = np.random.permutation(len(X_train))
X_train = X_train[train_indices]
y_train = y_train[train_indices]


# 转换为独热编码
y_train_onehot = np.zeros((y_train.shape[0], 2))
y_train_onehot[np.arange(y_train.shape[0]), y_train.flatten()] = 1

# 初始化模型
model = ManualCNN()
batch_size = 32
losses = []
accuracies = []

for epoch in range(20):
    epoch_loss = 0
    correct = 0

    # 随机打乱数据
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train_onehot[indices]

    for i in range(0, len(X_train), batch_size):
        # 获取小批量
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # 前向传播
        logits, loss = model.forward(X_batch, y_batch)
        epoch_loss += loss * X_batch.shape[0]

        # 反向传播
        model.backward(y_batch)

        # 计算准确率
        preds = np.argmax(logits, axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        correct += np.sum(preds == true_labels)

    # 统计指标
    avg_loss = epoch_loss / len(X_train)
    accuracy = correct / len(X_train)
    losses.append(avg_loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/20 | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

# 测试
test_logits = model.forward(X_test)
test_preds = np.argmax(test_logits, axis=1)
test_acc = np.mean(test_preds == y_test.flatten())
print(f"\nTest Accuracy: {test_acc:.4f}")

# 可视化
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Training Accuracy')
plt.legend()
plt.show()
np.random.seed(42)  # 固定随机选择
sample_indices = np.random.choice(len(X_test), 5, replace=False)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(sample_indices):
    # 获取图像和标签
    img = X_test[idx] + 0.5  # 反归一化到[0,1]
    true_label = "cat" if y_test[idx][0] == 0 else "dog"

    # 预测
    logits = model.forward(np.expand_dims(X_test[idx], axis=0))  # 添加batch维度
    pred_label = "cat" if np.argmax(logits) == 0 else "dog"

    # 可视化
    plt.subplot(1, 5, i + 1)
    plt.imshow(img.transpose(1, 2, 0))  # CHW -> HWC
    plt.title(f"True: {true_label}\nPred: {pred_label}", color='green' if true_label == pred_label else 'red')
    plt.axis('off')

plt.tight_layout()
plt.show()