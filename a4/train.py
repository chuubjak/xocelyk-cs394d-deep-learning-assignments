import torch
import numpy as np

from models import Detector, save_model
from utils import load_detection_data
import utils
try:
    from . import dense_transforms
except:
    import dense_transforms
import torch.utils.tensorboard as tb
from matplotlib import pyplot as plt
from models import extract_peak

# if torch.has_mps:
#     print('Device: mps')
#     device = torch.device('mps')
# else:
#     print('Device: cpu')
#     device = torch.device('cpu')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import torch.nn as nn

def augment(x, y):
    x, y = dense_transforms.ColorJitter(.5, .5, .5)(x, y)
    return x, y

def visualize_heatmaps(y_true, y_pred):

    with torch.no_grad():
        # Convert the input heatmap to a PyTorch tensor
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)



        # Move the heatmap to the device
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)

        # Move the heatmap to the CPU
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Stack the RGB channels
    true_heatmap = np.clip(y_true[0].transpose(1, 2, 0), 0, 1)
    pred_heatmap = np.clip(y_pred[0].transpose(1, 2, 0), 0, 1)

    # Plot the heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(true_heatmap)
    ax1.set_title("True Heatmap")
    ax1.axis("off")

    ax2.imshow(pred_heatmap)
    ax2.set_title("Predicted Heatmap")
    ax2.axis("off")

    plt.show()



def train(args):
    from os import path
    best_map = 0
    model = Detector()
    model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)


    pos_weight = torch.tensor([10])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    # Load the train and valid data with ToHeatmap transform
    print('loading train data')
    train_data = load_detection_data('../dense_data/train', transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),
                                                                           dense_transforms.ToTensor()]))
    print('loading valid data')
    valid_data = load_detection_data('../dense_data/valid', transform=dense_transforms.Compose([dense_transforms.ToTensor()]))
    num_epochs = 100
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
    # Run SGD for several epochs
    global_step = 0
    for epoch in range(num_epochs):
        print(epoch)
        train_loss = []

        batch_counter = 0
        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)
            x, y = augment(x, y)
            # print(y.shape)
            # assert 0 == 1
            batch_counter += 1
            if batch_counter % 100 == 0:
                print(f'Batch {batch_counter}/{len(train_data)}')
            optimizer.zero_grad()
            output = model(x)
            # print(y)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        if (epoch + 1) % 10 == 0:
            visualize_heatmaps(y, output)
        

        train_loss = np.mean(train_loss)
        scheduler.step(train_loss)

        print('Epoch [{}/{}], Loss: {:.4f}, Train MAP: {:.4f}, Valid MAP: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss, np.nan, np.nan))
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
