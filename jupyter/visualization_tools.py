import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
import seaborn as sns
from sklearn.metrics import confusion_matrix

from datasets import get_dataset
from models import STNet, STCoordNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name.split('_')[0] == 'stnet':
        model = STNet()
    elif model_name.split('_')[0] == 'stcoordconv':
        model = STCoordNet()
    else:
        logging.error('Invalid model')

    model.load_state_dict(torch.load(f'../models/{model_name}'))
    model.eval()
    model.to(device)

    dataset_name = model_name.split('_')[1]
    _, _, test_loader = get_dataset(dataset_name, root='../datasets')

    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


def visualize_mistakes(model_name, labels=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name.split('_')[0] == 'stnet':
        model = STNet()
    elif model_name.split('_')[0] == 'stcoordconv':
        model = STCoordNet()
    else:
        logging.error('Invalid model')

    model.load_state_dict(torch.load(f'../models/{model_name}'))
    model.eval()
    model.to(device)

    dataset_name = model_name.split('_')[1]
    _, _, test_loader = get_dataset(dataset_name, root='../datasets')

    with torch.no_grad():

        predictions = torch.Tensor().to(device)
        targets = torch.Tensor().to(device)

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]

            targets = torch.cat((targets, target))
            predictions = torch.cat((predictions, pred.squeeze()))
            wrong = ~target.eq(pred.squeeze())

            try:
                misclassified_images = torch.cat((misclassified_images, data[wrong, :, :, :]), 0)
            except NameError:
                misclassified_images = data[wrong, :, :, :]

        mistakes = ~targets.eq(predictions)

        print(f"Wrong predictions: {mistakes.sum().item()}/{len(test_loader.dataset)}")

        input_tensor = misclassified_images[:64, :, :, :].cpu()
        transformed_input_tensor = model.stn(misclassified_images[:64, :, :, :]).cpu()

        if labels is None:
            print('Predictions:')
            print(predictions[mistakes][:64].reshape(8, 8).cpu().numpy())
            print('Labels:')
            print(targets[mistakes][:64].reshape(8, 8).cpu().numpy())
        else:
            label_tags = {k: v for k, v in enumerate(labels)}
            print('Predictions:')
            print(np.array([label_tags[v] for v in predictions[mistakes][:64].cpu().numpy()]).reshape(8, 8))
            print('Labels:')
            print(np.array([label_tags[v] for v in targets[mistakes][:64].cpu().numpy()]).reshape(8, 8))

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


def visualize_confusion_matrix(model_name, labels=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name.split('_')[0] == 'stnet':
        model = STNet()
    elif model_name.split('_')[0] == 'stcoordconv':
        model = STCoordNet()
    else:
        logging.error('Invalid model')

    model.load_state_dict(torch.load(f'../models/{model_name}'))
    model.eval()
    model.to(device)

    dataset_name = model_name.split('_')[1]
    _, _, test_loader = get_dataset(dataset_name, root='../datasets')

    with torch.no_grad():

        predictions = torch.Tensor().to(device)
        targets = torch.Tensor().to(device)

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]

            targets = torch.cat((targets, target))
            predictions = torch.cat((predictions, pred.squeeze()))

        cm = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())
        if labels is None:
            sns.heatmap(cm, annot=True, fmt='d')
        else:
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)


def compare_results(dataset_name):
    route = '../results'
    results = pd.DataFrame(
        columns=['name', 'dataset', 'n_params', 'time', 'epoch', 'test_loss', 'correct', 'accuracy', 'date'])

    for f in [f for f in os.listdir(route) if f.endswith('csv')]:
        results = results.append(pd.read_csv(os.path.join(route, f)))

    results = results[results['dataset'] == dataset_name]

    plt.figure()

    for name in results['name'].unique():
        results_data = results[results['name'] == name]

        results_data_time_grouped = results_data.groupby('date').agg({'time': 'max', 'epoch': 'max'})
        results_data_time_grouped['time_per_epoch'] = \
            results_data_time_grouped['time'] / results_data_time_grouped['epoch']

        print("Specifications: {}\t\t\tparameters: {}\t\tavg time per epoch: {:10.2f} seconds".format(
            name, results_data['n_params'].values[0], results_data_time_grouped['time_per_epoch'].mean()))

        num_of_samples = results_data['date'].nunique()
        results_data_grouped = results_data.groupby(['epoch']).agg({'accuracy': ('mean', 'std')})
        label = f'{name} - nº of runs: {num_of_samples}'

        plt.errorbar(results_data_grouped.index, results_data_grouped['accuracy']['mean'],
                     yerr=results_data_grouped['accuracy']['std'], label=label, fmt='--o', capsize=6)
    plt.legend()
    plt.title(dataset_name)

    plt.show()
