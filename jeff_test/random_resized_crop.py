import math
import random
import torch
import torch.nn.functional as F


def resize(img, size, interpolation='bilinear'):
    r"""Resize the input tensor to the given size.
    Args:
        img (Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio.
        interpolation (str, optional): Desired interpolation. Default is
            ``'bilinear'``
    Returns:
        Tensor: Resized image.
    """
    # 确保输入是 3D tensor
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    
    if isinstance(size, int):
        w, h = img.shape[-1], img.shape[-2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return F.interpolate(img.unsqueeze(0), size=(oh, ow), mode=interpolation, align_corners=False).squeeze(0)
    else:
        return F.interpolate(img.unsqueeze(0), size=size[::-1], mode=interpolation, align_corners=False).squeeze(0)


def crop(img, i, j, h, w):
    """Crop the given tensor.
    Args:
        img (Tensor): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
    Returns:
        Tensor: Cropped image.
    """
    # 确保输入是 3D tensor
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    return img[:, i:i+h, j:j+w]


def resized_crop(img, i, j, h, w, size, interpolation='bilinear'):
    """Crop the given tensor and resize it to desired size.
    Args:
        img (Tensor): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner
        j (int): j in (i,j) i.e coordinates of the upper left corner
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (str, optional): Desired interpolation. Default is
            ``'bilinear'``.
    Returns:
        Tensor: Cropped image.
    """
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


class RandomResizedCrop:
    """Crop the given tensor to random size and aspect ratio.
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        """
        # 确保输入是 3D tensor
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
            
        height, width = img.shape[-2:]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback
        w = min(width, height)
        i = (height - w) // 2
        j = (width - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image to be cropped and resized.
        Returns:
            Tensor: Randomly cropped and resized image.
        """
        # 确保输入是 tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        
        # 确保在正确的设备上
        device = img.device
        
        # 获取参数
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        
        # 执行裁剪
        img = crop(img, i, j, h, w)
        
        # 执行缩放，确保输出大小一致
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
            
        img = F.interpolate(
            img.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # 如果输入是 2D，返回 2D
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
            
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ')'
        return format_string
