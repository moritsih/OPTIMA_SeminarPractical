import numpy as np
from typing import Hashable, Optional, Union, Sequence
import scipy.ndimage
import cv2
from typing import *
from monai.transforms import *
from monai.config.type_definitions import KeysCollection

labels_mapping = {255: 0,
                  0: 1,
                  80: 2,
                  160: 3}

class LayerPositionToProbabilityMap(MapTransform):
    def __init__(self, keys: Sequence, target_size,target_keys: Sequence = None):
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
            #img_filter = np.expand_dims(cv2.bilateralFilter(img[0], 10, 50, 50), 0)
            img_filter = img
            d[key] = img_filter
        return d

class ConvertToMultiChannelMasks(MapTransform):
    """
    Convert labels of ong image into 0,1,2,3
    0 - background
    1 - RNFL
    2 - GCIPL
    3 - Choroid
    """

    def __init__(self, keys: KeysCollection, target_keys: List[str], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.target_keys = target_keys


    def __call__(self, data):


        d = dict(data)
        for ki, key in enumerate(self.keys):

            mask = data[key]
            h, w= mask.shape

            mask_new = np.zeros((len(labels_mapping), h, w),dtype=np.uint8)
                      

            for i, (k, v) in enumerate(labels_mapping.items()):
                mask_new[i][mask==k] = 1

            d[self.target_keys[ki]] = mask_new[1:]
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

if __name__ == "__main__":
    import img
    transforms = Compose([
        LoadImaged(keys=['image','segmentation']),
        Lambdad(keys=['image','segmentation'], func = lambda x: x.transpose()[0:1]),
        #Lambdad(keys=['image','segmentation'], func = lambda x: np.expand_dims(x, 0)),
        #AddChanneld(keys=['image','segmentation']),
        RandZoomd(keys=["image", "segmentation"], mode=["area", "nearest-exact"], prob=0.3, min_zoom=1.3, max_zoom=1.3),
        Resized(keys=["image", "segmentation"], mode=["area", "nearest-exact"], spatial_size=[-1, 400]), # We first only resize horizontally, for the correct image width
        RandFlipd(keys=["image", "segmentation"], spatial_axis=1, prob=0.3),
        RandHistogramShiftd(keys=["image"], prob=0.3),
        RandAffined(keys=["image", "segmentation"], prob=0.3, shear_range=[(-0.7, 0.7), (0.0, 0.0)], translate_range=[(-300, 100), (0, 0)], mode=["bilinear", "nearest"], padding_mode="zeros"),
        Lambdad(keys=['image','segmentation'], func = lambda x: x[0, ...]),
        ConvertToMultiChannelMasks(keys=['segmentation'], target_keys=["masks"]),
        GetMaskPositions(keys=['masks'], target_keys=["mask_positions"]), #We get the layer position, but on the original height
        #AddChanneld(keys=['image','segmentation']),
        Resized(keys=["image", "segmentation", "masks"], mode=["area", "nearest-exact", "nearest-exact"], spatial_size=[400, 400]),
        Lambdad(keys=['mask_positions'], func = lambda x: x * 400 / 800), #We scale down the positions to have more accurate positions
        #Lambdad(keys=['image'], func = lambda x: np.clip((x - x.mean()) / x.std(), -1, 1)),
        Lambdad(keys=['image'], func = lambda x: 2*(x - x.min()) / (x.max() - x.min()) - 1 ),
        LayerPositionToProbabilityMap(["mask_positions"], target_size=(400,400), target_keys=["mask_probability_map"])
    ])

    #dd = {"image": "0001.png", 'segmentation': "0001m.png"}
    #hugy = transforms(dd)
    #print(hugy)
    #img.dump_image_normalized("trans-0001.png", hugy["image"][0])
    #img.dump_image_normalized("trans-0001m.png", hugy["segmentation"][0])