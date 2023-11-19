from .models import ClassificationLoss, model_factory, save_model, load_model
from .models import LinearClassifier
from .utils import accuracy, load_data
import torch


def train(args):
    model = model_factory[args.model]()
    loss_fn = ClassificationLoss()

    # Load the train and valid data
    train_data = load_data('data/train')
    valid_data = load_data('data/valid')
    print(args.model)
    if args.model == 'linear':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        num_epochs = 50
    elif args.model == 'mlp':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        num_epochs = 200
    # Run SGD for several epochs
    for epoch in range(num_epochs):
        for x, y in train_data:
            # reset the gradients
            optimizer.zero_grad()
            # forward pass
            # running model(x) automatically calls model.forward(x)
            output = model(x)
            # calculate loss
            loss = loss_fn(output, y)
            # calculate gradients on loss wrt params using backpropagation
            loss.backward()
            # update the parameters
            optimizer.step()

        # validate the model
        # should probably do this in tensorboard
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in valid_data:
                output = model(x)
                # calculate accuracy
                correct += accuracy(output, y)
                total += y.size(0)
            print('Accuracy of the model on the validation set: {} %'.format(100 * correct / total))
        save_model(model)
    
    # print progress
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
