import torch.optim
import torchvision.models
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
# select hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import dataset
train_dataset = torchvision.datasets.MNIST("./mnist_train", train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST("./mnist_test", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)
train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=64)
# import the model
resnet = torchvision.models.resnet18(pretrained=False)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet.fc = nn.Linear(512, 10)
resnet.to(device=device)
# create optimizer
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01)
# train nn
epoch = 50
# loss function
loss_fn = nn.CrossEntropyLoss()
for i in range(epoch):
    for data in train_data:
        img, target = data
        t2pil = torchvision.transforms.ToPILImage()
        m = t2pil(img[0])
        m.save("inputs/006.jpeg")
        img = img.to(device)
        target = target.to(device)
        output = resnet(img)
        result_loss = loss_fn(output, target)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
    with torch.no_grad():
        total_bingo_num = 0
        for data in test_data:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            output = resnet(img)
            bingo_num = (output.argmax(1) == target).sum()
            total_bingo_num += bingo_num
        accuracy = total_bingo_num / len(test_dataset)
        print("第{}次训练准确率为:{}".format(i+1, accuracy))
        torch.save(resnet, "./models/HWClassification_{}.pth".format(i+1))
        print("第{}次训练模型保存成功".format(i+1))