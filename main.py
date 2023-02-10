from __future__ import print_function
import torch
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .models import *
import torch.optim as optim
import divya_repo.utils as ut
from tqdm import tqdm


class Cifar10SearchDataset(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


class DataLoader():
    def __init__(self):
        self.cuda = ut.check_for_cuda()

    def transforms(self):
        train_transforms = A.Compose([
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                            fill_value=0.5, mask_fill_value=None),
            A.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
            A.RandomCrop(32, 32),
            ToTensorV2()
        ])

        # train_transforms_torch = transforms.compose([transforms.RandomCrop(32, padding=4)])

        test_transforms = A.Compose([
            A.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
            ToTensorV2()
        ])

        return train_transforms, test_transforms  # , train_transforms_torch

    def load_dataset(self):
        train_transforms, test_transforms = self.transforms()
        train = Cifar10SearchDataset(root='./data', train=True,
                                     download=True, transform=train_transforms)
        test = Cifar10SearchDataset(root='./data', train=False,
                                    download=True, transform=test_transforms)
        return train, test

    def return_loaders(self):
        train, test = self.load_dataset()
        dataloader_args = dict(shuffle=True, batch_size=32, num_workers=2, pin_memory=True) if self.cuda else dict(
            shuffle=True,
            batch_size=64)
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
        return train_loader, test_loader


train_loader, test_loader = DataLoader().return_loaders()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
EPOCHS = 20
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()


class Train():
    def __init__(self):
        self.train_losses = []
        self.train_acc = []

    def train_model(self, model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, target)
            self.train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id = {batch_idx} Accuracy = {100 * correct / processed:0.2f}')
            self.train_acc.append(100 * correct / processed)
        return sum(self.train_losses)/len(self.train_losses), sum(self.train_acc)/len(self.train_acc)


class Test():
    def __init__(self):
        self.test_losses = []
        self.test_acc = []

    def test_model(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        pred_epoch = []
        target_epoch = []
        data_epoch = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss = criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                if (len(pred_epoch) < 10):
                  b_target = target.cpu().numpy()
                  b_pred = pred.view_as(target).cpu().numpy()
                  for i in range(len(b_target)):
                    if(len(pred_epoch)<10):
                      if (b_target[i] != b_pred[i]):
                        pred_epoch.append(b_pred[i])
                        target_epoch.append(b_target[i])
                        data_epoch.append(data[i].cpu().numpy())
                self.test_losses.append(test_loss.item())
                self.test_acc.append(100. *( correct / len(data)))
        test_loss = sum(self.test_losses)/ len(test_loader.dataset)

        print("\n Test set: Avergae loss: {:4f}, Accuracy = {}/{}({:.2f}%)\n".format(test_loss, sum(self.test_acc),
                                                                                     len(test_loader.dataset),
                                                                                     (sum(self.test_acc) / len(
                                                                                         self.test_acc))))
        return test_loss, sum(self.test_acc)/len(self.test_acc), pred_epoch, target_epoch, data_epoch


def train_and_test_model():
    ut.print_model_summary(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train = Train()
    test = Test()
    train_losses_all_epochs=[]
    train_acc_all_epochs=[]
    test_losses_all_epochs=[]
    test_acc_all_epochs=[]

    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_loss, train_acc = train.train_model(model, device, train_loader, optimizer, epoch)
        train_losses_all_epochs.append(train_loss)
        train_acc_all_epochs.append(train_acc)
        test_loss, test_acc, pred, target, data = test.test_model(model, device, test_loader)
        test_losses_all_epochs.append(test_loss)
        test_acc_all_epochs.append(test_acc)
        print(train_losses_all_epochs, train_acc_all_epochs, test_losses_all_epochs, test_acc_all_epochs)
    return train_losses_all_epochs, train_acc_all_epochs, test_losses_all_epochs, test_acc_all_epochs,pred,target,data


def main():
  train_losses, train_acc, test_losses, test_acc, pred, target, data = train_and_test_model()

  ut.draw_train_test_acc_loss(train_losses, train_acc, test_losses, test_acc)
  ut.draw_misclassified_images(pred, target, data, "misclassified with resnet")
  ut.draw_gradcam_images(model,data, pred, target)
