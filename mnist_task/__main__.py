from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision

from ..ml_utils.misc import CurrentDir, get_settings
curdir = CurrentDir(__file__)
params = get_settings(curdir)

writer = SummaryWriter('runs')
"""
torch.Size([100, 1, 28, 28])
torch.Size([100, 32, 26, 26])
torch.Size([100, 32, 26, 26])
torch.Size([100, 64, 24, 24])
torch.Size([100, 64, 24, 24])
torch.Size([100, 64, 12, 12])
torch.Size([100, 128, 10, 10])
torch.Size([100, 128, 10, 10])
torch.Size([100, 128, 5, 5])
torch.Size([100, 128, 5, 5])
torch.Size([100, 3200])
torch.Size([100, 128])
torch.Size([100, 128])
torch.Size([100, 128])
torch.Size([100, 10])
torch.Size([100, 10])
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        # self.fc1 = nn.Linear(9216, 128)
        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
"""


def plot_conv_activation(model, dataset, device):
    layer = 'conv3'
    # Visualize feature maps
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.cpu().detach()

        return hook

    model.conv1.register_forward_hook(get_activation(layer))
    data = dataset[0][0].to(device)
    data.unsqueeze_(0)
    output = model(data)

    act = activation[layer].squeeze()
    height = 8
    width = int(np.ceil(len(act) / height))
    fig, axarr = plt.subplots(ncols=height, nrows=width)
    for i in range(width):
        for j in range(height):
            idx = i * height + j
            if idx < len(act):
                axarr[i][j].imshow(act[idx])
                axarr[i][j].axis('off')
    plt.axis('off')
    plt.savefig(f'{layer}')


def train(params, train_dataset, model, device, optimizer, scheduler):
    epoch = 0
    while True:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(np.floor(len(train_dataset) / params.folds)),
            shuffle=True)
        for eval_fold in range(params.folds):
            print(f"Epoch {epoch + 1}")
            model.train()
            for current_fold, (data, target) in enumerate(train_loader):
                if current_fold != eval_fold:
                    model.train()
                    optimizer.zero_grad()
                    for batch_idx in range(
                            0, int(np.floor(len(data) / params.batch_size)),
                            params.batch_size):
                        location = batch_idx * params.batch_size
                        batch_data = data[location:location +
                                          params.batch_size].to(device)
                        batch_target = target[location:location +
                                              params.batch_size].to(device)
                        output = model(batch_data)
                        loss = F.nll_loss(output, batch_target)
                        loss.backward()
                        optimizer.step()
                else:
                    eval_data = data
                    eval_target = target
            val_loss = 0
            for batch_idx in range(
                    0, int(np.floor(len(eval_data) / params.batch_size)),
                    params.batch_size):
                location = batch_idx * params.batch_size
                batch_data = eval_data[location:location +
                                       params.batch_size].to(device)
                batch_target = eval_target[location:location +
                                           params.batch_size].to(device)
                model.eval()
                output = model(batch_data)
                loss += F.nll_loss(output, batch_target).item()
            writer.add_scalar('val_loss', loss.item(), global_step=epoch)
            writer.add_histogram('conv1_bias',
                                 model.conv1.bias,
                                 global_step=epoch)
            writer.add_histogram('conv1_weight',
                                 model.conv1.weight,
                                 global_step=epoch)
            writer.add_histogram('conv1_weight_grad',
                                 model.conv1.weight.grad,
                                 global_step=epoch)
            print(f"Validation loss: {loss.item()}\n\n")
            epoch += 1
            if epoch == params.epochs:
                return
            else:
                scheduler.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    images, labels = next(iter(test_loader))
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images.to(device))
    print(
        f'\nTest set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset)})\n'
    )
    return test_loss


def main():
    file_name = curdir('mnist_cnn.pt')
    print(file_name)
    torch.manual_seed(params.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
    train_dataset = datasets.MNIST(
        '_datasets_',
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST('_datasets_',
                                  train=False,
                                  transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    model = Net().to(device)
    print(model)
    optimizer = optim.Adadelta(model.parameters(),
                               lr=params.lr)  # I read the adadelta paper
    scheduler = StepLR(optimizer, step_size=1, gamma=params.gamma)
    if not os.path.exists(file_name) or params.force_retrain:
        print('File not found, training')
        train(params, train_dataset, model, device, optimizer, scheduler)
        torch.save(model.state_dict(), curdir("mnist_cnn.pt"))
    else:

        print("File exists")
        model.load_state_dict(torch.load(file_name, map_location=device))
    test(model, device, test_loader)
    # plot(model, test_dataset, device)


# Test set: Average loss: 0.027601005390275143, Accuracy: 9905/10000 (99.05)

if __name__ == '__main__':
    main()
