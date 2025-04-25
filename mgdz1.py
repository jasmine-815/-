import matplotlib
import numpy as np
import cupy as cp  # 使用 CuPy 以支持 GPU 加速
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.ndimage import rotate  # 数据增强所需

from layers import MaxPool2D

matplotlib.use('TkAgg')  # 或 'Agg'，根据你的环境选择


# --------------------------
# 数据加载与预处理
# --------------------------
def load_dataset(data_dir, img_size=(32, 32), use_gpu=True):
    """
    加载数据集并进行预处理。

    参数:
        data_dir (str): 数据集目录路径。
        img_size (tuple): 图像大小 (宽, 高)。
        use_gpu (bool): 是否使用 GPU 加速。

    返回:
        tuple: 预处理后的图像和标签 (images, labels)。
    """
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            # 根据文件名解析标签：cat=0, dog=1
            label = 0 if filename.startswith('cat') else 1

            # 加载图像并进行预处理
            img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
            img = img.resize(img_size)  # 调整图像大小
            img_array = np.array(img).transpose(2, 0, 1)  # 转换为 CHW 格式
            img_array = img_array / 255.0 - 0.5  # 归一化到 [-0.5, 0.5]

            images.append(img_array)
            labels.append(label)

    # 根据是否使用 GPU 决定使用 NumPy 或 CuPy
    array_module = cp if use_gpu else np
    return (array_module.array(images, dtype=array_module.float32),
            array_module.array(labels, dtype=array_module.int32).reshape(-1, 1))


def augment_data(images):
    """
    数据增强：随机翻转和旋转图像。

    参数:
        images (ndarray): 输入图像。

    返回:
        ndarray: 增强后的图像。
    """
    augmented_images = []
    for img in images:
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1]  # 水平翻转
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            # 将 CuPy 数组转换为 NumPy 数组
            img_numpy = img.get() if isinstance(img, cp.ndarray) else img
            # 使用 SciPy 的 rotate 函数
            img_numpy = rotate(img_numpy, angle, axes=(1, 2), reshape=False)
            # 将 NumPy 数组转换回 CuPy 数组
            img = cp.array(img_numpy)
        augmented_images.append(img)
    return cp.array(augmented_images)


# --------------------------
# 优化的卷积层
# --------------------------
class Conv2D:
    """
    卷积层实现。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_gpu=True):
        self.array_module = cp if use_gpu else np
        self.kernel_size = kernel_size
        # 初始化卷积核和偏置
        self.kernels = self.array_module.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(self.array_module.float32) * 0.1
        self.bias = self.array_module.zeros((out_channels, 1), dtype=self.array_module.float32)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (ndarray): 输入数据，形状为 (B, C, H, W)。

        返回:
            ndarray: 输出数据。
        """
        self.input = x
        B, C, H, W = x.shape
        K = self.kernel_size
        OC = self.kernels.shape[0]
        out_h = H - K + 1
        out_w = W - K + 1

        # 获取输入的列形式
        patches = self._im2col(x, K, out_h, out_w)  # (B, IC * K * K, OH * OW)

        # 展平卷积核
        kernels_flat = self.kernels.reshape(OC, -1)  # (OC, IC * K * K)

        # 矩阵乘法
        output_flat = kernels_flat @ patches  # (B, OC, OH * OW)
        output_flat = output_flat.transpose(0, 2, 1)  # 调整维度

        # 调整输出形状并加上偏置
        output = output_flat.reshape(B, OC, out_h, out_w) + self.bias.reshape(OC, 1, 1)
        return output
    def backward(self, grad_output, lr=0.001):
        B, OC, OH, OW = grad_output.shape
        IC = self.kernels.shape[1]
        K = self.kernel_size

        grad_output_flat = grad_output.reshape(B, OC, -1)  # (B, OC, OH * OW)
        im2col_output = self._im2col(self.input, K, OH, OW)  # (B, IC * K * K, OH * OW)

        grad_kernels = self.array_module.zeros_like(self.kernels)
        grad_bias = self.array_module.zeros_like(self.bias)
        grad_input = self.array_module.zeros_like(self.input)

        for b in range(B):
            grad_kernel_flat = grad_output_flat[b] @ im2col_output[b].T  # (OC, IC * K * K)
            grad_kernel_reshaped = grad_kernel_flat.reshape(OC, IC, K, K)
            grad_kernels += grad_kernel_reshaped
            grad_bias += self.array_module.sum(grad_output[b],
                                               axis=(1, 2), keepdims=True).reshape(-1, 1)
            kernels_flat = self.kernels.reshape(OC, -1)  # (OC, IC * K * K)
            grad_input_flat = kernels_flat.T @ grad_output_flat[b]  # (IC * K * K, OH * OW)
            grad_input[b] = self._col2im(grad_input_flat, (1, *self.input[b].shape), K)

        self.kernels -= lr * grad_kernels / B
        self.bias -= lr * grad_bias / B
        return grad_input

    def _im2col(self, x, K, out_h, out_w):
        """
        将输入张量转化为列形式，用于卷积计算。

        参数:
            x (ndarray): 输入张量，形状为 (B, C, H, W)。
            K (int): 卷积核大小。
            out_h (int): 输出高度。
            out_w (int): 输出宽度。

        返回:
            ndarray: 转换后的列形式，用于矩阵乘法，形状为 (B, C * K * K, out_h * out_w)。
        """
        B, C, H, W = x.shape
        patches = self.array_module.zeros((B, C, K, K, out_h, out_w), dtype=x.dtype)
        for i in range(K):
            for j in range(K):
                patches[:, :, i, j, :, :] = x[:, :, i:i + out_h, j:j + out_w]
        # 转换为 (B, C * K * K, out_h * out_w)
        return patches.reshape(B, C * K * K, out_h * out_w)

    def _col2im(self, cols, input_shape, K):
        B, C, H, W = input_shape
        out_h = H - K + 1
        out_w = W - K + 1
        x = self.array_module.zeros(input_shape, dtype=cols.dtype)
        cols_reshaped = cols.reshape(C, K, K, B, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
        for i in range(K):
            for j in range(K):
                x[:, :, i:i + out_h, j:j + out_w] += cols_reshaped[:, :, i, j, :, :]
        return x


# --------------------------
# 其它组件
# --------------------------
class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Flatten:
    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.original_shape)


class Linear:
    def __init__(self, in_features, out_features, use_gpu=True):
        self.array_module = cp if use_gpu else np
        self.weights = (self.array_module.random.randn
                        (in_features, out_features).astype(self.array_module.float32) * 0.1)
        self.bias = self.array_module.zeros((1, out_features), dtype=self.array_module.float32)

    def forward(self, x):
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output, lr=0.001):
        grad_weights = self.input.T @ grad_output
        grad_bias = self.array_module.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.weights.T

        self.weights -= lr * grad_weights / grad_output.shape[0]
        self.bias -= lr * grad_bias / grad_output.shape[0]
        return grad_input


class SoftmaxCrossEntropy:
    def __init__(self, use_gpu=True):
        self.array_module = cp if use_gpu else np

    def forward(self, x, y):
        exps = self.array_module.exp(x - self.array_module.max(x, axis=1, keepdims=True))
        self.probs = exps / self.array_module.sum(exps, axis=1, keepdims=True)
        loss = -self.array_module.sum(y * self.array_module.log(self.probs + 1e-8)) / x.shape[0]
        return loss

    def backward(self, y):
        return (self.probs - y) / y.shape[0]


# --------------------------
# 学习率调度器
# --------------------------
class LearningRateScheduler:
    def __init__(self, initial_lr, decay_factor, decay_epoch):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_epoch = decay_epoch

    def get_lr(self, current_epoch):
        return self.initial_lr * (self.decay_factor ** (current_epoch // self.decay_epoch))


# --------------------------
# 定义优化后的 CNN 模型
# --------------------------
class ManualCNN:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.conv1 = Conv2D(3, 32, kernel_size=3, use_gpu=use_gpu)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, use_gpu=use_gpu)
        self.conv2 = Conv2D(32, 64, kernel_size=3, use_gpu=use_gpu)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, use_gpu=use_gpu)
        self.flatten = Flatten()
        self.fc1 = Linear(64 * 6 * 6, 128, use_gpu=use_gpu)
        self.relu_fc1 = ReLU()
        self.fc2 = Linear(128, 2, use_gpu=use_gpu)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x, y=None):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu_fc1.forward(x)
        x = self.fc2.forward(x)

        if y is not None:
            loss = self.loss_fn.forward(x, y)
            return x, loss
        return x

    def backward(self, y, lr):
        grad = self.loss_fn.backward(y)
        grad = self.fc2.backward(grad, lr)
        grad = self.relu_fc1.backward(grad)
        grad = self.fc1.backward(grad, lr)
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad, lr)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad, lr)
        return grad


# --------------------------
# 训练流程
# --------------------------
# 加载数据
X_train, y_train = load_dataset("dataset/train", img_size=(32, 32), use_gpu=True)
X_test, y_test = load_dataset("dataset/test", img_size=(32, 32), use_gpu=True)

# 数据增强
X_train = augment_data(X_train)

# 打乱训练数据
train_indices = cp.random.permutation(len(X_train))
X_train = X_train[train_indices]
y_train = y_train[train_indices]

# 转换为独热编码
y_train_onehot = cp.zeros((y_train.shape[0], 2))
y_train_onehot[cp.arange(y_train.shape[0]), y_train.flatten()] = 1

# 初始化模型和学习率调度器
model = ManualCNN()
scheduler = LearningRateScheduler(initial_lr=0.001, decay_factor=0.5, decay_epoch=5)
batch_size = 32
losses = []
accuracies = []

for epoch in range(20):  # 增加训练轮数
    lr = scheduler.get_lr(epoch)
    epoch_loss = 0
    correct = 0

    # 随机打乱数据
    indices = cp.random.permutation(len(X_train))
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
        model.backward(y_batch, lr)

        # 计算准确率
        preds = cp.argmax(logits, axis=1)
        true_labels = cp.argmax(y_batch, axis=1)
        correct += cp.sum(preds == true_labels)

    # 统计指标
    avg_loss = epoch_loss / len(X_train)
    accuracy = correct / len(X_train)
    losses.append(avg_loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/20 | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

# 测试
test_logits = model.forward(X_test)
test_preds = cp.argmax(test_logits, axis=1)
test_acc = cp.mean(test_preds == y_test.flatten())
print(f"\nTest Accuracy: {test_acc:.4f}")

losses = [l.get() if isinstance(l, cp.ndarray) else l for l in losses]
accuracies = [a.get() if isinstance(a, cp.ndarray) else a for a in accuracies]

# 可视化
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()