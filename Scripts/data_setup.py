from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch
def create_dataloaders(
    train_dir : str,
    test_dir : str,
    transform : transforms.Compose,
    batch_size : int,
    num_workers : int,
    sampler = False
) -> list[DataLoader, DataLoader]:
    '''
    Function for creating dataloaders
    Args:
        train_dir (str) : dir of training data
        test_dir (str) : dir of test data
        transform (transforms.Compose) : transform to perform on data
        batch_size (int) : Number of samples per batch 
        num_workers (int) : number of workers per dataloader
    '''

    # Using Imagefolder to load images
    # Images in seperate folders, one folder for each label
    train_data = datasets.ImageFolder(root = train_dir, 
                                      transform = transform
                                      )
    test_data = datasets.ImageFolder(root = test_dir, 
                                     transform = transform
                                     )

    # Load into data loadsers
    # If using weighted sampler
    if sampler == True:
 
        targets = torch.tensor(train_data.targets)  # Get the labels from the dataset
        class_counts = torch.bincount(targets)   # Count how many samples per class
        class_weights = 1.0 / class_counts.float()  # Inverse of the counts gives weights for each class


        sample_weights = class_weights[targets]  # Create weights for each sample

        # Create the WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_dataloader = DataLoader(train_data, 
                                    batch_size=batch_size,  
                                    num_workers = num_workers,
                                    pin_memory = True,
                                    sampler= sampler # Pin memory is use to allow easier transfer to gpu
                                    )
    else:
        train_dataloader = DataLoader(train_data, 
                                    batch_size=batch_size, 
                                    num_workers = num_workers,
                                    pin_memory = True,
                                    shuffle=True # Pin memory is use to allow easier transfer to gpu
                                    )
    test_dataloader = DataLoader(test_data, 
                                 batch_size=batch_size, 
                                 shuffle = True, 
                                 num_workers = num_workers, 
                                 pin_memory = True
                                 )

    return train_dataloader, test_dataloader
