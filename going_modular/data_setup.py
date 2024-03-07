import os
import torch
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()


def create_dataloader(
        train_dir: str,
        test_dir: str,
        transforms: transforms.Compose,
        batch_size: int,
        num_worker=0
):
    train_data = datasets.ImageFolder(train_dir, transform=transforms)
    test_data = datasets.ImageFolder(test_dir, transform=transforms)
    classes_name = train_data.classes
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_worker,
                                                   pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_worker,
                                                  pin_memory=True)
    return classes_name, train_dataloader, test_dataloader
