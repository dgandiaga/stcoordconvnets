import unittest
from pathlib import Path
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import logging

from models import STNet, ConvNet, CoordConvNet, STCoordNet
from models_birds import STNetBirds, ConvNetBirds, CoordConvNetBirds, STCoordNetBirds

from datasets import get_dataset


class TestDatasets(unittest.TestCase):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    dataset_names = ['mnist', 'fashion-mnist']
    model_names = ['convnet', 'stnet', 'coordconv', 'stcoordconv']

    def test_datasets_models_compatibility(self):

        for dataset in self.dataset_names:
            train_loader, val_loader, test_loader = get_dataset(dataset)
            data, target = next(iter(train_loader))

            self.assertEqual(data.shape[0], 64)
            self.assertEqual(data.shape[1], 1)
            self.assertEqual(data.shape[2], 28)
            self.assertEqual(data.shape[3], 28)

            for name in self.model_names:
                if name == 'stnet':
                    model = STNet()
                elif name == 'convnet':
                    model = ConvNet()
                elif name == 'coordconv':
                    model = CoordConvNet()
                elif name == 'stcoordconv':
                    model = STCoordNet()
                else:
                    logging.error(f'Please specify a valid model. Model specified: {name}')

                output = model(data)

                self.assertEqual(output.shape[0], target.shape[0])
                logging.info(f'Tested compatibility between {dataset} and {name}')

    def test_birds_compatibility(self):

        train_loader, val_loader, test_loader = get_dataset('birds')
        data, target = next(iter(train_loader))

        self.assertEqual(data.shape[0], 32)
        self.assertEqual(data.shape[1], 3)
        self.assertEqual(data.shape[2], 200)
        self.assertEqual(data.shape[3], 200)

        for name in self.model_names:
            if name == 'stnet':
                model = STNetBirds()
            elif name == 'convnet':
                model = ConvNetBirds()
            elif name == 'coordconv':
                model = CoordConvNetBirds()
            elif name == 'stcoordconv':
                model = STCoordNetBirds()
            else:
                logging.error(f'Please specify a valid model. Model specified: {name}')

            output = model(data)

            self.assertEqual(output.shape[0], target.shape[0])
            logging.info(f'Tested compatibility between birds and {name}')


if __name__ == '__main__':
    unittest.main()
