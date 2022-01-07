import torch
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
from torch import Tensor
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os
from ..ml_utils.summary_writer import SummaryWriter
from ..ml_utils.misc import CurrentDir
curdir = CurrentDir(__file__)


def cross_validation_splits(data, n_folds=10):
    np.random.shuffle(data)
    chunk_size = int(np.floor(len(data) / n_folds))
    total_length = chunk_size * n_folds
    folds = torch.unsqueeze(torch.from_numpy(
        np.transpose(np.array(np.array_split(data[:total_length], n_folds)),
                     axes=(0, 2, 1))),
                            dim=3).float()
    for i in range(len(folds)):
        yield folds[i], torch.cat((folds[0:i], folds[i + 1:len(folds)]))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 100  #40 best
        self.fc1 = Linear(1, n)
        self.fc3 = Linear(n, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc3(x)
        return x


class RegressionManager:
    def plot_model(self, net_model, test_loss):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_train = self.train_data[:, 0]
        y_train = self.train_data[:, 1]
        x_test = self.test_data[:, 0]
        y_test = self.test_data[:, 1]
        x_highres = torch.linspace(np.min(x_train), np.max(x_train),
                                   1000).to(device)
        net_output = net_model(x_highres)
        plt.close('all')
        plt.figure(figsize=(12, 8))
        plt.plot(x_test,
                 y_test,
                 color='b',
                 ls='',
                 marker='.',
                 label='Test data points')
        plt.plot(x_train,
                 y_train,
                 color='r',
                 ls='',
                 marker='.',
                 label='Train data points')
        plt.plot(x_highres.cpu().numpy(),
                 net_output,
                 color='g',
                 ls='--',
                 label='Network output (trained weights)')
        plt.title(f"Test loss: {test_loss}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(curdir('imgs/model_and_data.png'))

    def train(self):
        optimizer = SGD(self.model.parameters(), lr=0.01)  # lr=10**(-(deg+2)))
        print(optimizer)
        n_epochs = 11000
        epoch = 0
        while True:
            for val_set, train_set in cross_validation_splits(self.train_data):
                for batch_x, batch_y in train_set:
                    optimizer.zero_grad()
                    batch_x = Variable(batch_x).to(self.device)
                    batch_y = Variable(batch_y).to(self.device)
                    y_pred = self.model(batch_x)
                    loss = self.criterion(y_pred, batch_y)
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    y_pred = self.model(val_set[0].to(self.device))
                    loss = self.criterion(y_pred, val_set[1].to(self.device))
                    self.writer.add_scalar('val_loss',
                                           loss.data.cpu().item(),
                                           global_step=epoch)
                    if epoch % 100 == 0:
                        print(f"Epoch: {epoch} Val loss: {loss.data.cpu()}")

                if epoch == n_epochs:
                    torch.save(self.model.state_dict(), 'model.pt')
                    return
                epoch += 1
            self.writer.add_scalar('test_loss', self.test(), global_step=epoch)

    def test(self):
        self.model.eval()
        x = torch.unsqueeze(torch.from_numpy(self.test_data[:, 0]).to(
            self.device),
                            dim=1).float()
        y_pred = self.model(x)
        test_loss = self.criterion(
            y_pred,
            torch.unsqueeze(torch.from_numpy(self.test_data[:, 1]),
                            dim=1).float().to(self.device))
        return test_loss.data.cpu().item()

    def __init__(self):
        #if os.path.exists('runs'):
        #   shutil.rmtree('runs', ignore_errors=True)
        print(curdir('.'))
        self.writer = SummaryWriter(curdir('runs'), reset=True)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = MSELoss().to(self.device)
        with open(curdir('test_data.csv')) as f:
            self.test_data = np.array([[float(y) for y in x.split(',')]
                                       for x in f.read().split('\n')[1:-1]])
        with open(curdir('train_data.csv')) as f:
            self.train_data = np.array([[float(y) for y in x.split(',')]
                                        for x in f.read().split('\n')[1:-1]])
        plt.plot(self.test_data[:, 0],
                 self.test_data[:, 1],
                 color='b',
                 ls='',
                 marker='.',
                 label='Test data points')
        plt.plot(self.train_data[:, 0],
                 self.train_data[:, 1],
                 color='r',
                 ls='',
                 marker='.',
                 label='Train data points')
        plt.title(f"Raw Data")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(curdir('imgs/raw_data.png'))

        self.model = Net().to(self.device)
        print(self.model)
        if not os.path.exists(curdir('model.pt')):
            self.train()
        self.model.load_state_dict(torch.load('model.pt'))
        test_loss = self.test()
        self.plot_model(
            lambda data: self.model(torch.unsqueeze(data, 1).to(self.device)).
            cpu().detach().numpy(), test_loss)
        self.writer.flush()


if __name__ == '__main__':
    regression_manager = RegressionManager()
