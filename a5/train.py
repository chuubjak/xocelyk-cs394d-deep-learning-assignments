from .planner import Planner, save_model, load_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

# if torch.has_mps:
#     # print('Device: mps')
#     device = torch.device('mps')
# else:
#     print('Device: cpu')
#     device = torch.device('cpu')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(args):
    from os import path
    # model = Planner()
    model = load_model()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    

    model = model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    import inspect

    # transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    drive_data = load_data('drive_data', num_workers=4, transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),
                                                                           dense_transforms.ToTensor()]))
    test_data = load_data('drive_data', num_workers=4, transform=dense_transforms.Compose([dense_transforms.ToTensor()]))
    
    # split drive data into train and test data

    criterion = torch.nn.MSELoss().to(device)

    global_step = 0
    for epoch in range(100):
        # print(epoch)
        # model.train()
        epoch_train_loss_vals = []
        train_counter = 0
        for img, gt in drive_data:
            img, gt = dense_transforms.ColorJitter(0.7, 0.7, 0.7)(img, gt)
            train_counter += 1
            # if train_counter % 4 == 0:
            #     continue
            img = img.to(device)
            gt = gt.to(device)
            output = model(img)
            loss_val = criterion(output, gt)
            epoch_train_loss_vals.append(loss_val.item())
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        # model.eval()
        # # do test logging
        # epoch_test_loss_vals = []
        # test_counter = 0
        # with torch.no_grad():
        #     for img, gt in test_data:
        #         test_counter += 1
        #         if not test_counter % 4 == 0:
        #             continue
        #         img = img.to(device)
        #         gt = gt.to(device)
        #         output = model(img)
        #         loss_val = criterion(output, gt)
        #         epoch_test_loss_vals.append(loss_val.item())
        epoch_test_loss_vals = [0]

        scheduler.step(np.mean(epoch_train_loss_vals))
        print('Epoch: {}, Train Loss: {}, Test Loss: {}'.format(epoch, round(np.mean(epoch_train_loss_vals), 5), round(np.mean(epoch_test_loss_vals), 5)))
        if valid_logger is None or train_logger is None:
            pass
            # print('epoch %-3d' %
            #       (epoch))
        save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
