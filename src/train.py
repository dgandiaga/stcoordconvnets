# Python imports
import copy
import os
import time
from datetime import datetime
import logging
import argparse

# External imports
import matplotlib
import tqdm
import pandas as pd

# Pytorch imports
import torch
import torch.optim as optim
import torch.nn.functional as F

# Project imports
from models import STNet, ConvNet, CoordConvNet, STCoordNet
from models_birds import STNetBirds, ConvNetBirds, CoordConvNetBirds, STCoordNetBirds
from datasets import get_dataset


def train_model(name, dataset, epochs=30):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info(f'CUDA Availability: {torch.cuda.is_available()}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, _ = get_dataset(dataset)

    if dataset == 'birds':
        if name == 'stnet':
            model = STNetBirds().to(device)
        elif name == 'convnet':
            model = ConvNetBirds().to(device)
        elif name == 'coordconv':
            model = CoordConvNetBirds().to(device)
        elif name == 'stcoordconv':
            model = STCoordNetBirds().to(device)
        else:
            logging.error(f'Please specify a valid model. Model specified: {name}')
    else:
        if name == 'stnet':
            model = STNet().to(device)
        elif name == 'convnet':
            model = ConvNet().to(device)
        elif name == 'coordconv':
            model = CoordConvNet().to(device)
        elif name == 'stcoordconv':
            model = STCoordNet().to(device)
        else:
            logging.error(f'Please specify a valid model. Model specified: {name}')

    if dataset == 'mnist':
        optimizer = optim.SGD(model.parameters(), lr=0.005)
    elif dataset == 'fashion-mnist':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif dataset == 'birds':
        optimizer = optim.SGD(model.parameters(), lr=0.01)


    def train(epoch):
        model.train()
        logging.info(f'Epoch {epoch}')

        for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    def test():
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # sum up batch loss
                test_loss += F.nll_loss(output, target, size_average=False).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            return test_loss, correct

    test_results = pd.DataFrame(
        columns=['name', 'dataset', 'n_params', 'time', 'epoch', 'test_loss', 'correct', 'accuracy'])

    start = time.time()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    predicted = 0
    for epoch in range(1, epochs + 1):

        train(epoch)
        test_loss, correct = test()
        test_results.loc[len(test_results)] = [name, dataset, params, int(time.time() - start), epoch, test_loss,
                                               correct,
                                               correct / len(test_loader.dataset)]
        logging.info(f'Epoch {epoch} finished with test loss: {test_loss} and accuracy: '
                     f'{correct}/{len(test_loader.dataset)}')
        if correct > predicted:
            logging.info(f'Model accuracy improved from {predicted}/{len(test_loader.dataset)} to '
                         f'{correct}/{len(test_loader.dataset)}, saving current version of the model...')
            predicted = correct
            best_model = copy.deepcopy(model.state_dict())


    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_results['date'] = date

    torch.save(best_model, os.path.join(os.getcwd(), 'models', f'{name}_{dataset}_{date}.pt'))

    test_results.to_csv(os.path.join('results', f'{name}_{dataset}_{date}.csv'), index=False)

    return f'Model {name}_{dataset}_{date} trained and saved\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training loop')
    parser.add_argument('--dataset', required=True, choices=['mnist', 'fashion-mnist', 'birds'], help='dataset')
    parser.add_argument('--model', required=True, choices=['convnet', 'stnet', 'coordconv', 'coordstnet'], help='model architecture')
    parser.add_argument('--epochs', help='epoch number', type=int)
    args = parser.parse_args()
    train_model(args.model, args.dataset, args.epochs)