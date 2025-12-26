import torch
import torchvision
import torchvision.transforms as transforms

class dataset_loader():
    def __init__(self):
        self. cifar10_mean = (0.4914, 0.4822, 0.4465)
        self.cifar10_std = (0.2023, 0.1994, 0.2010)

    def load_data(self, directory, batchsize):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.cifar10_mean, self.cifar10_std)
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=directory, download=False, train=True, transform = self.transform)
        test_dataset = torchvision.datasets.CIFAR10(root=directory, download=False, train=False, transform = self.transform)

        trainloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=False, batch_size=batchsize, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batchsize, shuffle=False, num_workers=2
        )

        return trainloader, testloader