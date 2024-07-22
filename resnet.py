import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import ipdb
import yaml
from torchvision.models import ResNet18_Weights, ResNet34_Weights


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--data_path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda'], help='device to use for training')
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()
    # ipdb.set_trace()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(224),  # 将图像大小调整为 224x224
        transforms.ToTensor(),  # 将图像转换为 Tensor
    ])

    # 加载 CIFAR-10 数据集
    # trainset = torchvision.datasets.CIFAR10(root=args.data-path, train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 加载预训练的 ResNet-18 模型
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # 修改最后的全连接层以适应 CIFAR-10 数据集的 10 个类别
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # 使用指定设备进行训练（CPU 或 GPU）
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.learning_rate, momentum=0.9)

    # 训练模型
    if False:
        for epoch in range(args.epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

        print("Finished Training")

    # 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

if __name__ == '__main__':
    main()
