
from matplotlib import pyplot as plt


def loss_curve(train_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    # plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)

    # 保存为 SVG 格式
    plt.savefig('loss_curve.svg', format='svg')