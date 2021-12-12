import unittest
from pathlib import Path
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import logging

from models import STNet, ConvNet, CoordConvNet, STCoordNet
from models_birds import STResnextBirds, ResnextBirds
from datasets import get_dataset


class TestDatasets(unittest.TestCase):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_names = ['mnist', 'fashion-mnist']

    def test_datasets_models_compatibility(self):

        model_names = ['convnet', 'stnet', 'coordconv', 'stcoordconv']

        for dataset in self.dataset_names:
            train_loader, val_loader, test_loader = get_dataset(dataset)
            data, target = next(iter(train_loader))
            data, target = data.to(self.device), target.to(self.device)

            self.assertEqual(data.shape[0], 64)
            self.assertEqual(data.shape[1], 1)
            self.assertEqual(data.shape[2], 28)
            self.assertEqual(data.shape[3], 28)

            for name in model_names:
                if name == 'stnet':
                    model = STNet().to(self.device)
                elif name == 'convnet':
                    model = ConvNet().to(self.device)
                elif name == 'coordconv':
                    model = CoordConvNet().to(self.device)
                elif name == 'stcoordconv':
                    model = STCoordNet().to(self.device)
                else:
                    logging.error(f'Please specify a valid model. Model specified: {name}')

                output = model(data)

                self.assertEqual(output.shape[0], target.shape[0])
                logging.info(f'Tested compatibility between {dataset} and {name}')


if __name__ == '__main__':
    unittest.main()
