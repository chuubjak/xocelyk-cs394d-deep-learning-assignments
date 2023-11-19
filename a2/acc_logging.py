from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    for epoch in range(10):
        torch.manual_seed(epoch)
        epoch_train_acc = []
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = (epoch/10. + torch.randn(10)).mean()
            epoch_train_acc.append(dummy_train_accuracy)
            train_logger.add_scalar('loss', dummy_train_loss, global_step=epoch*20+iteration)
        epoch_train_acc = torch.tensor(epoch_train_acc).mean()
        train_logger.add_scalar('accuracy', epoch_train_acc, global_step=epoch*20+iteration)

        torch.manual_seed(epoch)
        epoch_validation_acc = []
        for iteration in range(10):
            dummy_validation_accuracy = (epoch/10. + torch.randn(10)).mean()
            epoch_validation_acc.append(dummy_validation_accuracy)
        epoch_validation_acc = sum(epoch_validation_acc)/len(epoch_validation_acc)
        # honestly no idea why the global step is logged that way but it works
        valid_logger.add_scalar('accuracy', epoch_validation_acc, global_step=(epoch + 1) * 20)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
