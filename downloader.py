import torch
import torchvision
import torchvision.transforms as transforms


# 1. Define transforms (Convert to Tensor and Normalize)
# Standard CIFAR-10 stats
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

# 2. Load the Training Set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
# 3. Load the Test Set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)

print(trainset)