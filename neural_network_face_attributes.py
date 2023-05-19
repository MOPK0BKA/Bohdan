import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import SGD
import time

# region Preparation

amount = len(glob.glob('data/celeba/*'))
batch_size = 1013
data_root = "data"
load_model = False

df = pd.read_csv('data/list_attr_celeba.csv')
list_attr = df[['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']]
list_attr = list_attr.to_numpy()
list_attr = torch.from_numpy(list_attr)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5]),
                                transforms.Resize(size=[64, 64])])

dataset = ImageFolder(root=data_root, transform=transform)

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size, )


# endregion

# region Secondary

######################################
#         Secondary functions        #
######################################

def save_checkpoint(state, filename="my_attr_easy.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("=> Load checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def show_image(img):
    img = img.detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()


def show_images(imgs, quantity):
    imgs = imgs.detach().numpy()
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    for j in range(quantity):
        plt.subplot(5, 5, j + 1)
        plt.imshow(imgs[j])
    plt.show()


def show_image_with_landmarks(img, array, numer_landmarks):
    img = img.detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    plt.imshow(img[0])
    plt.scatter(array[numer_landmarks * batch_size][:, 0], array[numer_landmarks * batch_size][:, 1], s=60)
    plt.show()


def show_images_with_landmarks(img, array, number_landmarks):
    img = img.detach().numpy()
    img = np.transpose(img, (0, 2, 3, 1))
    for j in range(25):
        plt.subplot(5, 5, j + 1)
        plt.imshow(img[j])
        plt.scatter(array[j + number_landmarks * batch_size][:, 0], array[j + number_landmarks * batch_size][:, 1], s=3)
        plt.axis('off')
    plt.show()


def result(model, imges, attr, idx):
    response = model(imges)
    coincidence = 0
    for i in range(40):
        if response[idx][i] > 0:
            response[idx][i] = 1
        else:
            response[idx][i] = -1
    for i in range(40):
        if response[idx][i] == attr[idx][i]:
            coincidence += 1
    print('Prawidlowo: ', coincidence, '  z 40')


def results(model, imges, attr, idx):
    response = model(imges)
    coincidence = 0
    for i in range(40):
        if response[idx][i] > 0:
            response[idx][i] = 1
        else:
            response[idx][i] = -1
    for i in range(40):
        if response[idx][i] == attr[idx][i]:
            coincidence += 1
    return coincidence * 100 / 40


def results_attr(model, imges, attr, quantity):
    response = model(imges)
    coincidence = torch.zeros(quantity, 40)
    percent = torch.zeros(40)
    for idx in range(quantity):
        for i in range(40):
            if response[idx][i] > 0:
                response[idx][i] = 1
            else:
                response[idx][i] = -1
        for i in range(40):
            if response[idx][i] == attr[idx][i]:
                percent[i] += 1
    for i in range(quantity):
        for j in range(40):
            percent[j] = percent[j] * 100 / quantity
    return percent


# endregion

# region Net
######################################
#           Net to training          #
######################################


class NET(nn.Module):
    in_channels = 3
    num_classes = 40
    linear_first_dimension = 2048

    def __init__(self, in_channels=in_channels, num_classes=num_classes, linear_first_dimension=linear_first_dimension):
        super(NET, self).__init__()
        self.NET1 = nn.Conv2d(in_channels, 8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.NET2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.NET3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(linear_first_dimension, num_classes)

    def forward(self, x):
        out = self.pool(self.NET1(x))
        out = self.pool(self.NET2(out))
        out = self.pool(self.NET3(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out


model = NET()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# endregion


# region Training
######################################
#             Training               #
######################################

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))


def train_model(model, optimizer, list_attr, train_loader, batch_size, n_epochs=50):
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
        y = torch.zeros(batch_size, 40)
        for batch_idx, (features, _) in enumerate(train_loader):
            if batch_idx == len(train_loader) - 1:
                break
            x = features.to()
            for i in range(batch_size):
                y[i] = list_attr[batch_idx * batch_size + i]
            loss_value = criterion(model(x), y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            if batch_idx % 40 == 0:
                print('\tPostępowanie epoche:', batch_idx / 2, '%')

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


epoch_data, loss_data = train_model(model, optimizer, list_attr, train_loader, batch_size)

# endregion

plt.plot(epoch_data, loss_data)
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (per batch)')

real_images = next(iter(train_loader))[0]

show_images(real_images, 25)

percent_correct = results(model, real_images, list_attr, 25)
print('Procent poprawnosci: ', format(percent_correct, '.3f'))

percentages_correct = results_attr(model, real_images, list_attr, 100)
print('Procent poprawnosci: ', format(percentages_correct, '.3f'))
