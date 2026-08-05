
import os 
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

def create_dataloaders(
    train_dir:str,
    test_dir:str,
    transform:transforms.Compose,
    batch_size:int,
    num_workers:int
):

train_data = datasets.ImageFolder(train_dir,transform=transform)
teat_data = datasets.ImageFolder(test_dir,transform=transform)

class_name=train_data.classes

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)
return train_dataloader,test_dataloader,class_name
