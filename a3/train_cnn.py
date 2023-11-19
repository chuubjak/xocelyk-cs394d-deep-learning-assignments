from models import CNNClassifier, save_model
from utils import ConfusionMatrix, load_data, LABEL_NAMES
from dense_transforms import *
import torch
import torchvision
import torch.utils.tensorboard as tb


def augment(x):
    # random horizontal flip batch of images with 50% probability
    if torch.rand(1) < 0.5:
        x = torchvision.transforms.functional.hflip(x)

    # another augmentation
    if torch.rand(1) < 0.5:
        x = torchvision.transforms.functional.adjust_brightness(x, 0.5)

    # another augmentation
    if torch.rand(1) < 0.5:
        x = torchvision.transforms.functional.adjust_contrast(x, 0.5)
    
    # another augmentation
    if torch.rand(1) < 0.5:
        x = torchvision.transforms.functional.adjust_saturation(x, 0.5)




    return x

def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    model = CNNClassifier()
    if torch.has_mps:
        print('Device: mps')
        device = torch.device('mps')
    else:
        print('Device: cpu')
        device = torch.device('cpu')

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Load the train and valid data
    train_data = load_data('../../data/train')
    valid_data = load_data('../../data/valid')
    num_epochs = 100
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    lr_logger = tb.SummaryWriter(path.join(args.log_dir, 'lr'))

    # Run SGD for several epochs
    for epoch in range(num_epochs):
        train_acc = None
        valid_acc = None
        train_loss = []

        for x, y in train_data:
            x = x.to(device)
            x = augment(x)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            print(output.shape)
            print(output[0, :])
            print(y.shape)
            print(y[0])
            loss = loss_fn(output, y)
            print(loss.item())
            print()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)

        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in train_data:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                correct += output.argmax(dim=1).eq(y).sum().item()
                total += y.size(0)

            train_logger.add_scalar('accuracy', 100 * correct / total, epoch)
            train_acc = 100 * correct / total

        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in valid_data:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                correct += output.argmax(dim=1).eq(y).sum().item()
                total += y.size(0)
            valid_logger.add_scalar('accuracy', 100 * correct / total, epoch)
            valid_acc = 100 * correct / total
        
        scheduler.step(train_loss)
        lr_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        print('Epoch [{}/{}], Loss: {:.4f}, Train Acc: {:.2f}%, Valid Acc: {:.2f}%'.format(epoch+1, num_epochs, train_loss, train_acc, valid_acc))
        # save_model(model)
    

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
