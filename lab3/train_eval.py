import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np


# 数据预处理和模型训练的代码
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train(
    model, train_loader, val_loader, num_epochs=100, patience=10, learning_rate=0.001
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_model_state_dict = model.state_dict()
    patience_count = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印每个batch的训练信息
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(train_loader)}] - Training Loss: {loss.item():.4f}"
                )

        model.eval()
        val_accuracy = []
        val_labels = []
        val_losses = []

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_accuracy.extend((predicted == labels).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss.item())

            mean_val_accuracy = np.mean(val_accuracy)
            mean_val_loss = np.mean(val_losses)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {mean_val_accuracy:.4f} - Validation Loss: {mean_val_loss:.4f}"
            )

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_val_accuracy = mean_val_accuracy
                best_model_state_dict = model.state_dict()
                patience_count = 0
                # 保存模型到磁盘
                torch.save(best_model_state_dict, "best_model.pth")
            else:
                patience_count += 1

            if patience_count >= patience:
                print(f"Early stopping after {patience} epochs of no improvement.")
                # 打印最好的模型的准确率和损失
                print(
                    f"Best Validation Accuracy: {best_val_accuracy:.4f} - Best Validation Loss: {best_val_loss:.4f}"
                )
                break

    model.load_state_dict(best_model_state_dict)
    return model


# 模型评估的代码
def eval(model, val_loader):
    # 模型评估
    model.eval()
    val_accuracy = []
    val_labels = []
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        val_accuracy.extend((predicted == labels).cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

    # accuracy = accuracy_score(val_labels, val_accuracy)
    accuracy = np.mean(val_accuracy)
    print(f"Test Accuracy: {accuracy:.4f}")
