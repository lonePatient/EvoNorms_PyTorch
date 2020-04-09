import torch
import argparse
import torch.nn as nn
from models.resnet import resnet18
from tools.progressbar import ProgressBar
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tools.trainingmonitor import TrainingMonitor
from tools.common import AverageMeter
from tools.common import seed_everything
loss_fn = nn.CrossEntropyLoss()

def data_loader(args):
    data = {
        'train': datasets.CIFAR10(
            root='./data', download=True,
            transform=transforms.Compose([
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )
        ),
        'valid': datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )
        )
    }
    loaders = {
        'train': DataLoader(data['train'], batch_size=args.batch_size, shuffle=True,
                            num_workers=10, pin_memory=True,
                            drop_last=True),
        'valid': DataLoader(data['valid'], batch_size=args.batch_size,
                            num_workers=10, pin_memory=True,
                            drop_last=False)
    }
    return loaders

def train(args,loaders,model):
    train_monitor = TrainingMonitor(file_dir='./logs/', arch=f'resnet18_{args.norm_type}_{args.batch_size}')
    train_loader,valid_loader = loaders['train'],loaders['valid']
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9,weight_decay=1e-4)
    for epoch in range(1,args.epochs + 1):
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        train_loss = AverageMeter()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            pbar(step=batch_idx, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        valid_log = evaluate(valid_loader, model)
        train_log = {'loss':train_loss.avg}
        logs = dict(train_log, **valid_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        print(show_info)
        train_monitor.epoch_step(logs)

def evaluate(valid_loader,model):
    pbar = ProgressBar(n_total=len(valid_loader),desc='Evaluating')
    valid_loss = AverageMeter()
    valid_acc = AverageMeter()
    model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            valid_loss.update(loss,n = data.size(0))
            valid_acc.update(correct, n=1)
            count += data.size(0)
            pbar(step=batch_idx)
    return {'valid_loss':valid_loss.avg,
            'valid_acc':valid_acc.sum /count}

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='CIFAR10')
    parser.add_argument("--model", type=str, default='ResNet50')
    parser.add_argument("--task", type=str, default='image')
    parser.add_argument("--epochs", default=50,type=int)
    parser.add_argument('--batch_size',default=256,type=int)
    parser.add_argument("--seed",default=42,type=int)
    parser.add_argument('--norm_type',default='bn',choices = ['bn','enb0','ens0'])
    args = parser.parse_args()
    seed_everything(args.seed)
    loaders = data_loader(args)
    model = resnet18(args.norm_type)
    device = torch.device("cuda")
    model.to(device)
    train(args,loaders,model)

