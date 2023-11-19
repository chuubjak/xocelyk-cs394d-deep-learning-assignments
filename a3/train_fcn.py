import torch
import numpy as np

from models import FCN, save_model
from utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, load_data
try:
    from . import dense_transforms
except:
    import dense_transforms
import torch.utils.tensorboard as tb

# PYTORCH_ENABLE_MPS_FALLBACK=1

if torch.has_mps:
    print('Device: mps')
    device = torch.device('mps')
else:
    print('Device: cpu')
    device = torch.device('cpu')

def augment(x, y):
    x, y = dense_transforms.RandomHorizontalFlip()(x, y)
    x, y = dense_transforms.ColorJitter(.5, .5, .5)(x, y)
    return x, y

def train(args):
    from os import path
    model = FCN()
    model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    best_valid_acc = 0

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1-w for w in DENSE_CLASS_DISTRIBUTION]).to(device))

    # Load the train and valid data
    train_data = load_dense_data('../dense_data/train')
    valid_data = load_dense_data('../dense_data/valid')
    num_epochs = 100
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, verbose=True)

    # Run SGD for several epochs
    global_step = 0
    for epoch in range(num_epochs):
        train_acc = None
        valid_acc = None
        train_loss = []

        batch_counter = 0
        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)
            print(x.shape, y.shape)
            batch_counter += 1
            if batch_counter % 100 == 0:
                print(f'Batch {batch_counter}/{len(train_data)}')
            y = y.long()
            new_x = []
            new_y = []
            for img, label in zip(x, y):
                img, label = augment(img, label)
                new_x.append(img)
                new_y.append(label)
            x = torch.stack(new_x)
            y = torch.stack(new_y)
            optimizer.zero_grad()
            output = model(x)
            print(output.shape, y.shape)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        train_loss = np.mean(train_loss)

        with torch.no_grad():
            train_acc_list = []
            for x, y in train_data:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                # detach output to cpu
                output = output.detach().cpu()
                y = y.detach().cpu()
                # Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
                # the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
                conf_matrix = ConfusionMatrix()
                conf_matrix.add(output.argmax(1), y)
                train_acc = conf_matrix.iou * 100
                train_acc_list.append(train_acc)
            train_acc = np.mean(train_acc_list)
            log(train_logger, x, y, output, global_step)
            global_step += 1

        with torch.no_grad():
            valid_acc_list = []
            for x, y in valid_data:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                output = output.detach().cpu()
                y = y.detach().cpu()
                # Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
                # the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
                conf_matrix = ConfusionMatrix()
                conf_matrix.add(output.argmax(1), y)
                valid_acc = conf_matrix.iou * 100
                valid_acc_list.append(valid_acc)
            valid_acc = np.mean(valid_acc_list)


        scheduler.step(valid_acc)

        train_logger.add_scalar('acc', train_loss, epoch)
        valid_logger.add_scalar('acc', valid_acc, epoch)
        print('Epoch [{}/{}], Loss: {:.4f}, Train IOU: {:.2f}%, Valid IOU: {:.2f}%'.format(epoch+1, num_epochs, train_loss, train_acc, valid_acc))
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            print('save model')
            save_model(model)
    

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
