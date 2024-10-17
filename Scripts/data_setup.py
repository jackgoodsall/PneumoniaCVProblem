from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
import torch
import numpy as np
import random
import os
from PIL import Image




def extract_patient_ids(filename):
    patient_id = filename.split('_')[0].replace("person", "")
    return patient_id



def split_file_names(input_folder, val_split_perc):
    # Pneumonia files contain patient id, so we group split them by patient to avoid data leakage
    pneumonia_patient_ids = set([extract_patient_ids(fn) for fn in os.listdir(os.path.join(input_folder, 'PNEUMONIA')) if fn.lower().endswith(('.jpeg', '.jpg', '.png'))])
    pneumonia_val_patient_ids = random.sample(list(pneumonia_patient_ids), int(val_split_perc * len(pneumonia_patient_ids)))

    pneumonia_val_filenames = []
    pneumonia_train_filenames = []

    for filename in os.listdir(os.path.join(input_folder, 'PNEUMONIA')):
        if not filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            continue
        patient_id = extract_patient_ids(filename)
        if patient_id in pneumonia_val_patient_ids:
            pneumonia_val_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))
        else:
            pneumonia_train_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))

    # Normal (by file, no patient information in file names)
    normal_filenames  = [os.path.join(input_folder, 'NORMAL', fn) for fn in os.listdir(os.path.join(input_folder, 'NORMAL')) if fn.lower().endswith(('.jpeg', '.jpg', '.png'))]
    normal_val_filenames = random.sample(normal_filenames, int(val_split_perc * len(normal_filenames)))
    normal_train_filenames = list(set(normal_filenames)-set(normal_val_filenames))

    train_filenames = pneumonia_train_filenames + normal_train_filenames
    val_filenames = pneumonia_val_filenames + normal_val_filenames

    return train_filenames, val_filenames



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





def create_dataloaders_with_validation_set(
    train_dir : str,
    test_dir : str,
    train_transform : transforms.Compose,
    test_transform : transforms.Compose,
    batch_size : int,
    num_workers : int,
    validation_split : float = 0.2,
    sampler = False
    ) -> list[DataLoader, DataLoader ,DataLoader]:
    '''
    Function for creating dataloaders
    Args:
        train_dir (str) : dir of training data
        test_dir (str) : dir of test data
        train_transform (transforms.Compose) : transform to perform on training  datra
        test_transform (transforms.Compose) : transform for test and validation data
        batch_size (int) : Number of samples per batch 
        validation_split (float) : Split of validation data
        num_workers (int) : number of workers per dataloader
    Returns:
        3 data loaders
    '''

    train_file_names, val_file_names = split_file_names(train_dir, validation_split)

    # Using Imagefolder to load images
    # Images in seperate folders, one folder for each label
    train_data = datasets.ImageFolder(root = train_dir, 
                                      transform = train_transform,
                                      is_valid_file= lambda x : x in train_file_names
                                      )
    
    validation_data = datasets.ImageFolder(root = train_dir,
                                           transform=test_transform,
                                           is_valid_file= lambda x : x in val_file_names)

    test_data = datasets.ImageFolder(root = test_dir, 
                                     transform = test_transform
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
    
    val_dataloader = DataLoader(validation_data, 
                                 batch_size=batch_size, 
                                 shuffle = True, 
                                 num_workers = num_workers, 
                                 pin_memory = True
                                 )

    return train_dataloader,val_dataloader, test_dataloader
