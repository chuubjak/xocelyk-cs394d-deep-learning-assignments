import torch
import torch.nn as nn
try:
    from .models import TCN, save_model
except:
    from models import TCN, save_model
try:
    from .utils import SpeechDataset, one_hot, vocab
except:
    from utils import SpeechDataset, one_hot, vocab
import numpy as np
from torch.utils.data import DataLoader


def idx_to_str(tens):
    return ''.join([vocab[i] for i in tens])

def make_random_batch(batch_size, seq_len, one_hot):
    B = []
    # for i in range(batch_size):
    s = np.random.choice(one_hot.size(1) - seq_len, batch_size, replace=True)
    for si in s:
        B.append(one_hot[:, si:si+seq_len])
    return torch.stack(B, dim=0)

def train(args):
    # torch plateau scheduler
    model = TCN()
    train_logger, valid_logger = None, None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'train'))
    #     valid_logger = tb.SummaryWriter(os.path.join(args.log_dir, 'valid'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)

    # Load the train and valid data
    train_data = SpeechDataset('../data/train.txt', transform=one_hot)
    valid_data = SpeechDataset('../data/valid.txt', transform=one_hot)

    batch_size = 32
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            y = x[:, :, :]
            y = y.to(device)

            optimizer.zero_grad()
            o = model(x)
            o = o[:, :, :-1]
            y = x.argmax(dim=1)
            # print(idx_to_str(o[0].argmax(dim=0)))
            loss = loss_fn(o, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # print(loss.item())

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}')
        # train_logger.add_scalar('loss', avg_train_loss, epoch)

        # Validation
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch_idx, x in enumerate(valid_loader):
                x = x.to(device)
                y = x[:, :, :]
                y = y.to(device)

                o = model(x)
                o = o[:, :, :-1]
                y = x.argmax(dim=1)
                loss = loss_fn(o, y)
                total_valid_loss += loss.item()

        # Calculate average validation loss
        avg_valid_loss = total_valid_loss / len(valid_loader)
        scheduler.step(avg_valid_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_valid_loss}')
        # valid_logger.add_scalar('loss', avg_valid_loss, epoch)
        save_model(model)

    # Save the model
    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
