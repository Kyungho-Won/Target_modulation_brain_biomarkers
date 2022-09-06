import os
import argparse
import datetime as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from dataloader import HISDataset

from model import EnsembleNet

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='dataset/total_images')
parser.add_argument('--csv_name', type=str, default='dataset/all.csv')
parser.add_argument('--class_category', type=str, default='head')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--bsize', type=int, default=64)
parser.add_argument('--random_split', dest='random_split', default=False, action='store_true')
parser.add_argument('--ensemble', dest='ensemble', default=False, action='store_true')
parser.add_argument('--save_model', type=str, default='./checkpoint')
parser.add_argument('--end_epoch', type=int, default=700)
args = parser.parse_args()

option = args.class_category + '_randomSplit' + str(args.random_split) \
    + '_ensemble' + str(args.ensemble)
model_save_path = os.path.join(args.save_model, option)
if not os.path.isdir(args.save_model):
    os.makedirs(args.save_model)
if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

if args.ensemble:
    net = EnsembleNet()
else:
    net = torchvision.models.resnet18(num_classes=2)
#print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    net = net.cuda()
    criterion = criterion.cuda()

transforms = tf.Compose(
    [
        tf.Resize(256),
        tf.RandomResizedCrop(args.input_size),
        #tf.RandomHorizontalFlip(),
        tf.ToTensor(),
    ]
)

loader_params = {'batch_size': args.bsize,
                 'shuffle': True,
                 'num_workers': 2,
                 'drop_last': True}

val_loader_params = {'batch_size': args.bsize,
                 'shuffle': False,
                 'num_workers': 2,
                 'drop_last': False}

if args.random_split:
    dataset = HISDataset(args, transform=transforms, train=True)
    train_dataset, val_dataset = random_split(dataset, [27000, len(dataset) - 27000])
else:
    args.csv_name = 'dataset/train.csv'
    train_dataset = HISDataset(args, transform=transforms, train=True)
    args.csv_name = 'dataset/val.csv'
    val_dataset = HISDataset(args, transform=transforms, train=True)
print("train_set: {}, val_set: {}".format(len(train_dataset), len(val_dataset)))

train_dataloader = DataLoader(dataset=train_dataset, **loader_params)
val_dataloader = DataLoader(dataset=val_dataset, **val_loader_params)

for epoch in range(0, args.end_epoch):
    net.train()
    avg_loss = 0
    correct = 0
    num_data = 0
    for batch in train_dataloader:     
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if args.cuda:
            input = input.cuda()
            target = target.cuda()
        if not args.ensemble:
            output = net(input)
        else:
            emotion = Variable(batch[2])
            emotion = emotion.cuda()
            output = net(input, emotion)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_dataloader)
        correct += torch.sum(torch.argmax(output, dim=1) == target).item()
        num_data += target.size(0)
    print("(train)===> Epoch[{}/{}]): Loss: {} Acc: {}".format(epoch+1, 
        args.end_epoch, avg_loss, correct)) #:.5f
    save_name = 'latest_model.pth'
    torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(model_save_path, save_name))

    correct = 0
    num_data = 0
    if epoch % 10 == 0:
        save_name = 'model_' + str(epoch) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(model_save_path, save_name))

        print("{} saved!".format(os.path.join(model_save_path, save_name)))

        for batch in val_dataloader:
            input, target = Variable(batch[0], requires_grad=False), \
                Variable(batch[1], requires_grad=False)
            if args.cuda:
                input = input.cuda()
                target = target.cuda()
            if not args.ensemble:
                output = net(input)
            else:
                emotion = Variable(batch[2], requires_grad=False)
                emotion = emotion.cuda()
                output = net(input, emotion)
            correct += torch.sum(torch.argmax(output, dim=1) == target).item()
            num_data += target.size(0)
        score = correct / num_data
        print("")
        print("(validation)===> Epoch[{}/{}]): Acc: {}".format(epoch+1, args.end_epoch, score))
        print("(validation)===> cor: {}, num: {}".format(correct, num_data))
        print("")



