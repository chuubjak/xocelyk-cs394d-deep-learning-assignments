from models import CNNClassifier, save_model, load_model
from utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb



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
    train_data = load_data('../data/train')
    valid_data = load_data('../data/valid')
    num_epochs = 1000
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run SGD for several epochs
    for epoch in range(num_epochs):
        train_acc = None
        valid_acc = None
        train_loss = None

        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

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
        
        print('Epoch [{}/{}], Loss: {:.4f}, Train Acc: {:.2f}%, Valid Acc: {:.2f}%'.format(epoch+1, num_epochs, train_loss, train_acc, valid_acc))
        save_model(model)
    

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
