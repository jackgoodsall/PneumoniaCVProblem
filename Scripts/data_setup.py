from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
    train_dir : str,
    test_dir : str,
    transform : transforms.Compose,
    batch_size : int,
    num_workers : int
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

    # Load data into DataLoaders
    train_dataloader = DataLoader(train_data, 
                                  batch_size=batch_size, 
                                  shuffle = True, 
                                  num_workers = num_workers,
                                  pin_memory = True # Pin memory is use to allow easier transfer to gpu
                                  )
    test_dataloader = DataLoader(test_data, 
                                 batch_size=batch_size, 
                                 shuffle = True, 
                                 num_workers = num_workers, 
                                 pin_memory = True
                                 )

    return train_dataloader, test_dataloader
