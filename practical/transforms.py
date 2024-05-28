from monai.transforms import *
from monai.config.type_definitions import KeysCollection

import numpy as np
import torch

from typing import *
from pathlib import Path
import cv2
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
import os

np.random.seed(99)
torch.manual_seed(99)


labels_mapping = {0: 0,
                  50: 1,
                  100: 2,
                  150: 3}


class ConvertToMultiChannelMasks(MapTransform):
    """
    Convert labels of image into 0,1,2,3
    0 - background
    50 - IRF
    100 - SRF
    150 - PED
    """

    def __init__(self, keys: KeysCollection, target_keys: List[str], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_keys = target_keys


    def __call__(self, data):

        assert len(self.keys) == len(self.target_keys), "Number of keys and target keys must be the same"
        assert len(mask.shape) == 3, "Mask must be 3D. If not, check 'ConvertLabelMaskToChannel' transform"

        d = dict(data)
        for key in self.keys:

            mask = data[key]
            c, h, w= mask.shape

            mask_new = np.zeros((len(labels_mapping), h, w),dtype=np.uint8)
                      
            for i, (k, v) in enumerate(labels_mapping.items()):
                mask_new[i][mask==k] = 1

            d[self.target_keys[i]] = mask_new[1:]
        return d
    

class ConvertToMultiChannelGOALS(MapTransform):
    """
    Convert labels of ong image into 0,1,2,3
    0 - background
    1 - RNFL
    2 - GCIPL
    3 - Choroid
    """

    def __call__(self, data):


        d = dict(data)
        for key in self.keys:

            mask = data[key]

            mask_new = np.zeros_like(mask,dtype=np.uint8)

            for k,v in labels_mapping.items():
                mask_new[mask==k] = v

            d[key] = mask_new 
        return d
    

class GetMaskPositions(MapTransform):
    def __init__(self, keys: KeysCollection, target_keys: List[str], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_keys = target_keys


    def __call__(self, data):


        d = dict(data)
        for ki, key in enumerate(self.keys):

            mask = data[key]
            num, h, w = mask.shape

            mask_positions = np.zeros((num * 2, w),dtype=np.float32)
                      

            for i in range(num):
                mask_positions[i*2] = np.argmax(mask[i], axis=0)
                mask_positions[i*2 + 1] = h - np.argmax(np.flip(mask[i], axis=0), axis=0)

            # The first and the second masks share a common border
            removal_mask = np.ones(len(mask_positions), dtype=bool)
            removal_mask[1] = False
            mask_positions = mask_positions[removal_mask]
            mask_positions = np.expand_dims(mask_positions, 1)
            d[self.target_keys[ki]] = mask_positions
            d["invalid_masks"] = np.ones_like(mask_positions)
        return d
    

class LayerPositionToProbabilityMap(MapTransform):
    def __init__(self, keys: Sequence, target_size, target_keys: Sequence = None):
        super().__init__(keys)
        if target_keys is None:
            self.target_keys = keys
        self.target_keys = target_keys
        self.target_size = target_size
    
    def smoothing_function(self, mask):
        mean = 0
        std = 0.5
        scale = 1 / (std * np.sqrt(2 * np.pi))
        return scale * np.exp(-(mask - mean)**2 / (2 * std**2))
        #return -(mask - mean)**2 / (2 * std**2)
        
    def __call__(self, data):
        for i, key in enumerate(self.keys):
            layer = data[key]
            column = np.arange(self.target_size[1])
            column = np.expand_dims(column, 1)
            mask = np.repeat(column, self.target_size[0], axis=1)
            mask = np.expand_dims(mask, 0)
            mask = np.repeat(mask, layer.shape[0], axis=0)
            #layer = np.expand_dims(layer, 1)
            #print(mask.shape, layer.shape)
            mask = mask - layer
            #mask = np.ones_like(mask)
            mask = mask.astype(np.float32)
            
            mask = self.smoothing_function(mask)
            mask = mask / np.expand_dims(mask.sum(axis=1), axis=2)
            data[self.target_keys[i]] = mask

        return data
    

class CropImages(MapTransform):

    def __init__(self, keys: KeysCollection, source_key : str, crop_size, crop_allowance=10, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.crop_size = crop_size
        self.crop_allowance = crop_allowance


    def __call__(self, data):


        d = dict(data)
        for ki, key in enumerate(self.keys):
            preliminary = data[self.source_key]
            min_val = max(preliminary.min() - self.crop_allowance, 0)
            img = data[key]
            img_crop = img[:, min_val:(min_val + self.crop_size), :]

            d[key] = img_crop
        return d


class CropValImages(MapTransform):

    def __init__(self, keys: KeysCollection, source_key : str, crop_size, crop_allowance=10, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.crop_size = crop_size
        self.crop_allowance = crop_allowance
        self.positions = [0, 1, 3, 5, 7]

    def __call__(self, data):


        d = dict(data)
        for ki, key in enumerate(self.keys):
            crop_id = data[self.source_key]
            img = data[key]
            img_crop = img[:, :, (self.positions[crop_id]*100):(self.positions[crop_id]*100 + self.crop_size)]

            d[key] = img_crop
        return d


class BilateralFilter(MapTransform):

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)


    def __call__(self, data):
        d = dict(data)
        for ki, key in enumerate(self.keys):
            img = data[key]
            img_filter = np.expand_dims(cv2.bilateralFilter(img[0], 10, 50, 50), 0)
            #img_filter = img
            d[key] = img_filter
        return d
    



class ExpandChannelDim(MapTransform):
    
        def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
            super().__init__(keys, allow_missing_keys)
    
    
        def __call__(self, data):
            d = dict(data)
            for ki, key in enumerate(self.keys):
                if isinstance(data[key], np.ndarray):
                    img = torch.from_numpy(data[key])
                else:
                    img = data[key]
                img = img.unsqueeze(0)
                d[key] = img
            return d


class TransposeImage(MapTransform):
    
        def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
            super().__init__(keys, allow_missing_keys)
    
    
        def __call__(self, data):
            d = dict(data)
            for ki, key in enumerate(self.keys):
                img = data[key]
                img = img.permute(0, 2, 1)
                d[key] = img
            return d
        

class SVDNA(MapTransform, Randomizable):

    def __init__(self, 
                 keys: KeysCollection, 
                 histogram_matching_degree: float = 0.5,
                 allow_missing_keys: bool = False, 
                 prob: float = 1.0,
                 source_domains: List = ["Spectralis", "Topcon", "Cirrus"], 
                 data_path: Path = Path(Path.cwd() / 'data/Retouch-Preprocessed/train')) -> None:
        super().__init__(keys, allow_missing_keys)

        self.prob = np.clip(prob, 0.0, 1.0)
        self.source_domains = source_domains
        self.named_domain_folder = Path.cwd() / 'data/RETOUCH/TrainingSet-Release' # path holding the img folders sorted by domain
        self.target_dataset = self.filter_target_domain(data_path, source_domains)
        self.domains = ['Spectralis', 'Topcon', 'Cirrus']
        self.histogram_matching_degree = histogram_matching_degree

    def __call__(self, data):
        d = dict(data)
        for ki, key in enumerate(self.keys):
            img = data[key]
            img = self.perform_SVDNA(img, self.target_dataset, self.source_domains, self.histogram_matching_degree)

            d[key] = img
        return d

    def filter_target_domain(self, data_path, source_domains):
        '''
        data_path: Path to the training set folder where all images are not sorted by domains.
        source_domains: The source domain for the upcoming SVDNA process.

        Returns: a dictionary containing three lists of dictionaries of the following structure:
                    {source domain: [{img: img1, label: label1}, {img: img2, label: label2}, ...], 
                    target domain 1: [{img: img1}, {img: img2}, ...],
                    target domain 2: [{img: img1}, {img: img2}, ...]}
        '''

        domains = os.listdir(self.named_domain_folder) # gets only the names of domains
        # creates dict e.g. {'cirrus':['path1', 'path2', ...], 'topcon':['path1', 'path2', ...]}
        img_folders_sorted_by_domain = {domain:os.listdir(self.named_domain_folder / domain) for domain in domains}

        # restructure source data into a list of dictionaries, where each dictionary has keys img and label

        unsorted_img_folders_training_set = os.listdir(data_path)

        domains_dict = {}

        for domain in domains:
            
            list_of_dicts_images = []
            
            for img_folder in unsorted_img_folders_training_set:

                if img_folder in img_folders_sorted_by_domain[domain]:
                    
                    subfolders = os.listdir(data_path / img_folder)
                    unsorted_img_folders_training_set.remove(img_folder)

                    if 'image' in subfolders and 'label_image' in subfolders:
                        
                        if domain in source_domains:
                            continue
                        
                        else:
                            sliced_images = sorted(os.listdir(data_path / img_folder / 'image'))

                            for i in range(len(sliced_images)):
                                    list_of_dicts_images.append({'img': str(data_path / img_folder / 'image' / sliced_images[i])})
                        
            domains_dict[domain] = list_of_dicts_images
                                        
        return domains_dict
    

    def perform_SVDNA(self, source, target_dataset, source_domains, histogram_matching_degree):
        '''
        Function takes an img from source domain and applied SVDNA to it.
        In order to do that, we have to sample 2 things for each picture:

        One of ['Spectralis', 'Topcon', 'Cirrus'] to know where to take the style from.
            if the source domain is chosen, no style transfer (i. e. svdna) is performed
        k some number between 20 and 50

        '''
        
        domain_idx = self.R.randint(len(self.domains))
        domain = self.domains[domain_idx]

        if domain not in source_domains:
            # randomly sample k and target image to get style from
            k = self.R.randint(20,50)
            
            target = target_dataset[domain][self.R.randint(0, len(target_dataset[domain]))]['img']

            source_img_raw, target_img_raw, _, _, img_svdna, source_noise_adapt_no_histogram = self.svdna(k, 
                                                                                                          target, 
                                                                                                          source, 
                                                                                                          histogram_matching_degree)
            #print("SVDNA performed.")

        else:
            img_svdna = cv2.imread(source, 0)
            #print("Source domain chosen. No SVDNA performed.")

        return img_svdna
    

    def readIm(self, imagepath):
        image = imagepath#cv2.imread(str(imagepath), 0)
        return image


    def svdna(self, k, target_img_path, source_img_path, histo_matching_degree):

        h, w = 1024, 496

        target_img = cv2.imread(target_img_path, 0)
        source_img = cv2.imread(source_img_path, 0)

        resized_target=np.asarray(Image.fromarray(target_img).resize((h, w), Image.NEAREST))
        resized_src=np.asarray(Image.fromarray(source_img).resize((h, w), Image.NEAREST))

        #print("A")
        u_target, s_target, vh_target = np.linalg.svd(resized_target, full_matrices=False)
        #print("target done")
        u_source, s_source, vh_source = np.linalg.svd(resized_src, full_matrices=False)
        #print("source done")
        thresholded_singular_target = s_target
        thresholded_singular_target[0:k] = 0
        #print("B")
        thresholded_singular_source = s_source
        thresholded_singular_source[k:] = 0

        # i thought the k last singular values of target would be transferred to the source. However, here
        # they just take them out, put the matrices back together and only THEN perform the subtraction

        target_style = np.array(np.dot(u_target, np.dot(np.diag(thresholded_singular_target), vh_target)))
        content_src = np.array(np.dot(u_source, np.dot(np.diag(thresholded_singular_source), vh_source)))

        content_trgt = target_img - target_style

        noise_adapted_im = content_src + target_style

        noise_adapted_im_clipped = noise_adapted_im.clip(0, 255).astype(np.uint8)
        #print("C")
        transformHist = A.Compose([
        A.HistogramMatching([resized_target], blend_ratio=(histo_matching_degree, histo_matching_degree), read_fn=self.readIm, p=1)
        ])
        #print("D")
        transformed = transformHist(image=noise_adapted_im_clipped)
        svdna_im = transformed["image"]

        return source_img, target_img, content_src, np.squeeze(target_style), svdna_im, noise_adapted_im_clipped
        

class Debugging(MapTransform):
        def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
            super().__init__(keys, allow_missing_keys)
        
        def __call__(self, data):
            for key in self.keys:
                if key == "masks":
                    print(
                          key, ":", 
                          "\nshape: ", data[key].shape, 
                          "\nmax: ", data[key][0].max(), 
                          "\nmin: ", data[key][0].min(), 
                          "\nmean: ", data[key][0].mean(), 
                          "\ndtype: ", data[key].dtype
                          )
            
            return data


class PrintMeanImageValues(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        
    def __call__(self, data):
        for key in self.keys:
            print(key, data[key].max())
        return data


class CustomImageLoader(MapTransform):

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        #empty = False
        for ki, key in enumerate(self.keys):
            #if "empty" in data[key]:
            #    empty = True
            d[key] = cv2.imread(data[key], cv2.IMREAD_GRAYSCALE)
            #print(d[key].shape, d[key].max(), d[key].min(), d[key].dtype) if empty else None
        return d
    

class ConvertLabelMaskToChannel(MapTransform):
    """
    Convert labels of image into color channels
    """

    def __init__(self, keys: KeysCollection, target_keys: List[str], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

        self.target_keys = target_keys
        self.pixel_class_map = {0: 0,
                                50: 1, 
                                100: 2, 
                                150: 3}
        

    def __call__(self, data):


        d = dict(data)
        for ki, key in enumerate(self.keys):

            mask = data[key]

            assert len(mask.shape) == 2, "Label must be 2D"

            h, w = mask.shape

            mask_new = np.zeros((len(self.pixel_class_map), h, w), dtype=np.uint8)
                      
            for i, (k) in enumerate(self.pixel_class_map.keys()):
                mask_new[i][mask==k] = 255

            d[self.target_keys[ki]] = mask_new
        
        return d
    

class ImageVisualizer(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):

        fig, ax = plt.subplots(1, len(self.keys), figsize=(20, 5))
        for i, key in enumerate(self.keys):
            image = data[key]

            if image.shape[0] == 1:
                image = np.squeeze(image, 0)
                ax[i].imshow(image, cmap='gray')
                ax[i].set_title(key)
            elif image.shape[0] == 4:
            #elif image.shape[0] == 3:
                print("only showing segmentation maps")
                image = np.transpose(image[1:, :, :], (1, 2, 0))
                #image = np.transpose(image, (1, 2, 0))
                ax[i].imshow(image)
                ax[i].set_title(key)
        
        plt.show()

        return data


class NormalizeToZeroOne(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key].float() / 255
        return data


