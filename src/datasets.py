import logging
from torchvision import datasets, transforms
import torch
from six.moves import urllib


def get_dataset(name, root='datasets'):

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    if name == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=64, shuffle=True, num_workers=4)
        # Test and validation dataset
        test_val_dataset = datasets.MNIST(root=root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        val_len = int(len(test_val_dataset)/2)
        val_dataset, test_dataset = torch.utils.data.random_split(test_val_dataset,
                                                                  [val_len, len(test_val_dataset)-val_len],
                                                                  generator=torch.Generator().manual_seed(42))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    elif name == 'fashion-mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root=root, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ])), batch_size=64, shuffle=True, num_workers=4)
        # Test and validation dataset
        test_val_dataset = datasets.FashionMNIST(root=root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        val_len = int(len(test_val_dataset) / 2)
        val_dataset, test_dataset = torch.utils.data.random_split(test_val_dataset,
                                                                  [val_len, len(test_val_dataset) - val_len],
                                                                  generator=torch.Generator().manual_seed(42))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    else:
        logging.error(f'Please specify a valid dataset. Dataset specified: {name}')




    return train_loader, val_loader, test_loader
