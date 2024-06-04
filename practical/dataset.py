import os
import cv2
import numpy as np
from pathlib import Path
from typing import Sequence, List
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
import numpy as np
import cv2
from tqdm import tqdm

np.random.seed(99)
torch.manual_seed(99)


SOURCE = 'Spectralis'
DOMAINS = ['Spectralis', 'Topcon', 'Cirrus']
TARGETS = [domain for domain in DOMAINS if domain != SOURCE]


class OCTDatasetPrep(Dataset):
    '''
    This class prepares the dataset for the SVDNA process. It filters the source domain and splits the dataset into training, validation and test sets.
    It also generates black images for missing label images in the label_image folder.

    Args:
        data_path (str): Path to the training set folder where all images are not sorted by domains.
        generate_empty_labels (bool, optional): Flag to generate black images for missing files in the label_image folder. Defaults to False.
        source_domains (List, optional): The source domain for the upcoming SVDNA process. Defaults to ['Spectralis', 'Topcon', 'Cirrus'].
        named_domain_folder (str, optional): Path to the folder where the img folders are sorted by domain. Defaults to 'data/RETOUCH/TrainingSet-Release'.
        get_decompositions (bool, optional): Flag to perform SVD on the images. Defaults to False.
        delete_decompositions (bool, optional): Flag to delete the decompositions. Defaults to False.

    Raises:
        Exception: Raises an exception if the label images are missing.
    '''
    
    def __init__(
                 self, 
                 data_path: str, 
                 generate_empty_labels=False, 
                 source_domains: List = ['Spectralis', 'Topcon', 'Cirrus'], 
                 named_domain_folder: str = 'data/RETOUCH/TrainingSet-Release',
                 ):

        self.data_path = Path(data_path)
        self.testset_path = Path.cwd() / 'data/Retouch-Preprocessed/test'

        self.named_domain_folder = Path.cwd() / named_domain_folder # path holding the img folders sorted by domain

        self.domains = os.listdir(self.named_domain_folder) # gets only the names of domains
        self.source_domains = source_domains

        if generate_empty_labels:
            self.generate_black_images() 

        try:
            self.source_domain_dict, self.num_domains = self.filter_source_domain(self.data_path, self.source_domains, self.domains, self.named_domain_folder)
        except IndexError:
            raise Exception("LABEL IMAGES MISSING! Have you tried generating all missing label images?")
        
        self.source_domain_list = [folder for domain in self.source_domains for folder in self.source_domain_dict[domain]]

    def __len__(self):
        return len(self.source_domain_list)


    def filter_source_domain(self, data_path, source_domains, domains, named_domain_folder):
        '''
        data_path: Path to the training set folder where all images are not sorted by domains.
        source_domains: The source domain for the upcoming SVDNA process.
        '''

        # creates dict e.g. {'cirrus':['path1', 'path2', ...], 'topcon':['path1', 'path2', ...]}
        img_folders_sorted_by_domain = {domain:os.listdir(named_domain_folder / domain) for domain in domains}

        # restructure source data into a list of dictionaries, where each dictionary has keys img and label

        unsorted_img_folders_training_set = os.listdir(data_path)

        domains_dict = {}

        for domain in domains:

            # gonna rewrite
            
            list_of_dicts_images = []
            
            for img_folder in unsorted_img_folders_training_set:

                if img_folder in img_folders_sorted_by_domain[domain]:
                    
                    subfolders = os.listdir(data_path / img_folder)
                    unsorted_img_folders_training_set.remove(img_folder)

                    if 'image' in subfolders and 'label_image' in subfolders:
                        
                        if domain in source_domains:

                            sliced_images = sorted([x for x in os.listdir(data_path / img_folder / 'image') if ".png" in x])
                            sliced_labels = sorted([x for x in os.listdir(data_path / img_folder / 'label_image') if ".png" in x])
                            
                            for i in range(len(sliced_images)):
                                
                                #if (sliced_images[i] == sliced_labels[i]):# or (sliced_images[i][:-4] + '_empty.png' == sliced_labels[i]):
                                if sliced_images[i] in sliced_labels:
                                    list_of_dicts_images.append(
                                        {'img': str(data_path / img_folder / 'image' / sliced_images[i]), 'label': str(data_path / img_folder / 'label_image' / sliced_images[i])}
                                        )

                                else:
                                    continue
                        
            domains_dict[domain] = list_of_dicts_images
                                        
        return domains_dict, len(domains)


    def generate_black_images(self, delete_images=False):
        """
        Generate black images for missing files in the label_image folder.

        Args:
            main_folder (str or Path): Path to the main folder.
            delete_images (bool, optional): Flag to delete the generated black images. Defaults to False.
        """
        main_folder = self.data_path

        if not delete_images:
        # Iterate over the subfolders
            for subfolder in sorted(os.listdir(main_folder)):
                subfolder_path = main_folder / subfolder

                # Check if the subfolder contains the 'image' and 'label_image' folders
                if os.path.isdir(subfolder_path) and 'image' in os.listdir(subfolder_path) and 'label_image' in os.listdir(subfolder_path):
                    image_folder = subfolder_path / 'image'
                    label_folder = subfolder_path / 'label_image'

                    # Get the set of filenames in the 'image' folder
                    image_files = sorted(set([x for x in os.listdir(image_folder) if ".png" in x]))

                    # Get the set of filenames in the 'label_image' folder
                    label_files = sorted(set([x for x in os.listdir(label_folder) if ".png" in x]))

                    # Find the filenames that are in 'label_image' but not in 'image'
                    missing_files = [i for i in image_files if i not in label_files]

                    # Create a black image for each missing file
                    for file in missing_files:
                        file_path = label_folder / file

                        # find the shape of the input image to create corresponding target
                        try:
                            file_shape = cv2.imread(str(image_folder / file), 0).shape
                        except AttributeError:
                            print(f"Error reading file: {file}")

                        black_image = np.zeros(file_shape, dtype='uint8')

                        # make unique names so files can be deleted again
                        cv2.imwrite(f"{str(file_path)[:-4]}_empty.png", black_image)

        # Delete the generated black images if delete_images flag is True
        if delete_images:
            for subfolder in sorted(os.listdir(main_folder)):
                subfolder_path = main_folder / subfolder
                if os.path.isdir(subfolder_path) and 'label_image' in os.listdir(subfolder_path):
                    label_folder = subfolder_path / 'label_image'
                    for file in os.listdir(label_folder):
                        file_path = label_folder / file
                        if "_empty" in str(file):
                            os.remove(str(file_path))

    def delete_generated_labels(self):
        # Delete the generated black images
        self.generate_black_images(delete_images=True)


    def get_test_dataset(self, testset_path):

        test_data_list = []
        
        for img_folder in os.listdir(testset_path):

            subfolders = os.listdir(testset_path / img_folder)

            if 'image' in subfolders and 'label_image' in subfolders:

                sliced_images = sorted(os.listdir(testset_path / img_folder / 'image'))
                sliced_labels = sorted(os.listdir(testset_path / img_folder / 'label_image'))
                
                #for i in range(len(sliced_images)):
                #    if (sliced_images[i] == sliced_labels[i]) or (sliced_images[i][:-4] + '_empty.png' == sliced_labels[i]):

                #        test_data_list.append(
                #            {'img': str(testset_path / img_folder / 'image' / sliced_images[i]), 'label': str(testset_path / img_folder / 'label_image' / sliced_labels[i])}
                #            )
                        
                for i in range(len(sliced_images)):

                    test_data_list.append(
                        {'img': str(testset_path / img_folder / 'image' / sliced_images[i]), 'label': str(testset_path / img_folder / 'label_image' / sliced_images[i])}
                        )
    
        return test_data_list



    def get_datasets(self, dataset_split: Sequence[float] = [0.7, 0.2, 0.1], use_official_testset=False):
        '''
        Returns the datasets in the order:
        training_set, validation_set, test_set
        '''

        dataset_len = len(self.source_domain_list)


        if use_official_testset:
            test_set = self.get_test_dataset(self.testset_path)
            dataset_split = [dataset_split[0], 1-dataset_split[0]]

            val_len = int(dataset_len * dataset_split[1])
            train_len = dataset_len - val_len

            self.training_set, self.validation_set = random_split(self.source_domain_list, [train_len, val_len])
            self.test_set = test_set
            print("Total dataset length: ", dataset_len + len(test_set))
            print(f"Training set: {len(self.training_set)}")
            print(f"Validation set: {len(self.validation_set)}")
            print(f"Test set: {len(self.test_set)}")

            return self.training_set, self.validation_set, self.test_set


        train_len = int(dataset_len * dataset_split[0])
        val_len = int(dataset_len * dataset_split[1])
        test_len = dataset_len - train_len - val_len

        self.training_set, self.validation_set, self.test_set = random_split(self.source_domain_list, [train_len, val_len, test_len])
        print("Total dataset length: ", dataset_len)
        print(f"Training set: {len(self.training_set)}")
        print(f"Validation set: {len(self.validation_set)}")
        print(f"Test set: {len(self.test_set)}")
        return self.training_set, self.validation_set, self.test_set



class MakeDataset(Dataset):

    def __init__(self, dataset_untransformed, transform=None):

        self.dataset_untransformed = dataset_untransformed
        self.transform = transform

    def __len__(self):
        return len(self.dataset_untransformed)
    
    def __getitem__(self, idx):
        sample = self.dataset_untransformed[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
