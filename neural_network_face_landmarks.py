import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torchvision 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torchvision import datasets, models, transforms

# region Preparation

amount = len(glob.glob('data/celeba/*'))
batch_size = 1013
data_root = "data"
load_model = False
reduction = 4

df = pd.read_csv('data/list_landmarks_align_celeba.csv')
b = df[['lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x',
        'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']]
a = b.to_numpy()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5]),
                                transforms.Resize(size=[int(218 / reduction), int(178 / reduction)]),
                                ])
dataset = ImageFolder(root=data_root, transform=transform)
dataset = ImageFolder(root=data_root, transform=transform)

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size)

# region Secondary

def save_checkpoint(state, filename="my_landmarks.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("=> Load checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def show_image(img, idx):
    img = img.detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    plt.imshow(img[idx])
    plt.show()


def show_image_with_landmarks(img, array, numer_landmarks):
    array = array.reshape(202599, 5, 2)
    img = img.detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    plt.imshow(img[2])
    plt.scatter(array[2 + numer_landmarks * batch_size][:, 0], array[2 + numer_landmarks * batch_size][:, 1], s=60)
    plt.show()


def show_images(img, quantity):
    img = img.detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    for j in range(quantity):
        plt.subplot(5, 5, j + 1)
        plt.imshow(img[j])
        # plt.axis('off')
    plt.show()


def show_images_with_landmarks(img, array, number_landmarks):
    array = array.reshape(202599, 5, 2)
    img = img.detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    for j in range(25):
        plt.subplot(5, 5, j + 1)
        plt.imshow(img[j])
        plt.scatter(array[j + number_landmarks * batch_size][:, 0], array[j + number_landmarks * batch_size][:, 1], s=3)
        plt.axis('off')
    plt.show()


def show_images_landmarks(img, array):
    array = array.reshape(1013, 5, 2)
    array = array.detach().numpy()
    img = img.detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    for j in range(25):
        plt.subplot(5, 5, j + 1)
        plt.imshow(img[j])
        plt.scatter(array[j][:, 0], array[j][:, 1], s=3)
        plt.axis('off')
    plt.show()


def show_image_landmarks(img, array):
    array = array.reshape(1013, 5, 2)
    img = img.detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    plt.imshow(img[2])
    plt.scatter(array[2][:, 0], array[2][:, 1], s=60)
    plt.show()


# endregion


# region Landmarks

landmarks_numpy = a.reshape(202599, 5, 2)


def rot_mat(deg):
    theta = deg / 180 * np.pi
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


ra = landmarks_numpy
angle = 0

for k in range(amount):
    if k % batch_size == 0 and not (k == 0):
        if angle == 5:
            angle = -1
        angle += 1
    for i in range(5):
        r = rot_mat(-angle * 15)
        rs = landmarks_numpy[k][i]
        rw = np.matmul(r, rs)
        ra[k][i] = rw
        ra[k][i][0] = int(ra[k][i][0] / reduction)
        ra[k][i][1] = int(ra[k][i][1] / reduction)
print('\nRozmiar landmarks:', ra.shape, '\n')

ra = ra.reshape(202599, 10)
tensor_for_numpy = torch.from_numpy(ra)
landmarks = tensor_for_numpy


# endregion



# endregion


# region Net
######################################
#          Net to training       #
######################################


class NET(nn.Module):
    in_channels = 3
    num_classes = 10
    linear_first_dimension = 960
    linear_second_dimension = int(linear_first_dimension / 2)

    def __init__(self, in_channels=in_channels, num_classes=num_classes,
                 linear_first_dimension=linear_first_dimension, linear_second_dimension=linear_second_dimension):
        super(NET, self).__init__()
        self.NET1 = nn.Conv2d(in_channels, 8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.NET2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.NET3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(linear_first_dimension, linear_second_dimension)
        self.fc2 = nn.Linear(linear_second_dimension, num_classes)

    def forward(self, x):
        out = self.pool(self.NET1(x))
        out = self.pool(self.NET2(out))
        out = self.pool(self.NET3(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = NET()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# endregion


# region Trenowanie
######################################
#            Trenowanie              #
######################################

if load_model:
    load_checkpoint(torch.load("my_landmarks.pth.tar"))


def train_model(model, optimizer, landmarks, train_loader, batch_size, n_epochs=50):
    start_time = time.time()
    # Optimization
    criterion = nn.MSELoss()
    # Train model
    losses = []
    epochs = []
    loss_value = 0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = n_epochs
        y = torch.zeros(batch_size, 10)
        rotation = 0
        for batch_idx, (features, _) in enumerate(train_loader):
            if batch_idx == len(train_loader) - 1:
                break
            x = features.to()
            transform = T.RandomRotation(degrees=(rotation, rotation))
            rotation = rotation + 15
            if rotation == 90:
                rotation = 0
            x = transform(x)
            for i in range(batch_size):
                y[i] = landmarks[batch_idx * batch_size + i]
            loss_value = criterion(model(x), y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            if batch_idx % 40 == 0:
                print('\tPostępowanie epoche:', batch_idx / 2, '%')
            if batch_idx == 3:
                show_image_landmarks(x, y)
                show_images_landmarks(x, model(x))

        print('Epoch: ', epoch, 'Loss: ', format(loss_value.item(), '.3f'))
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
        epochs.append(epoch / N)
        losses.append(loss_value.item())
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
        if loss_value.item() < 0.3:
            break

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    return np.array(epochs), np.array(losses)


epoch_data, loss_data = train_model(model, optimizer, landmarks, train_loader, batch_size)

# endregion


plt.plot(epoch_data, loss_data)
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (per batch)')

real_images = next(iter(train_loader))[0]
show_images_landmarks(real_images, model(real_images))

