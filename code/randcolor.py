from functools import wraps
import numpy as np
import cv2
# https://github.com/ZFTurbo/volumentations
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional.py#L876
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_pil.py#L68
# https://github.com/ZhenZHAO/AugSeg/blob/main/augseg/dataset/augs_TIBA.py#L185
# https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html#torchvision.transforms.ColorJitter
# https://github.com/yyliu01/TraCoCo/blob/main/Code/VnetLA/Dataloader/augmentation.py#LL25C4-L25C4


MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


def preserve_shape(func):
    """
    Preserve shape of the image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


@clipped
def _brightness_contrast_adjust_non_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.dtype
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


@preserve_shape
def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = np.dtype("uint8")

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = np.arange(0, max_value + 1).astype("float32")

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += beta * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img


def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)

    return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)



class RandomBrightnessContrast(object):
    def __init__(self, 
                 brightness_limit=0.5,
                 contrast_limit=0.5,
                 prob=0.8):
        
        self.contrast_limit = contrast_limit
        self.brightness_limit = brightness_limit
        
        self.alpha = 1.0
        self.beta = 0.0
        self.prob = prob
    
    def _random_update(self):
        self.alpha = 1.0 + np.random.uniform(-1.0 * self.contrast_limit, self.contrast_limit),
        self.beta = 0.0 + np.random.uniform(-1.0 * self.brightness_limit, self.brightness_limit)
        

    def __call__(self, image):
        image = image.astype(np.float32)
        self._random_update()
        if np.random.uniform() < self.prob:
            img_min, img_max = image.min(), image.max()
            image_norm = (image - img_min) / (img_max - img_min)
            image_norm = brightness_contrast_adjust(image_norm, alpha=self.alpha, beta=self.beta)
            image = image_norm * (img_max - img_min) + img_min

        return image