import numpy as np
import torch, torchvision
import torchvision.transforms as transforms

class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    taken from https://github.com/uoguelph-mlrg/Cutout
    Args:
        holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, holes, length):
        self.holes = holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_Transforms(img_height, img_width, mode = "weak"):
    if mode.lower() == "none":
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_height,img_width)),
            transforms.ToTensor(),
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))        
        ]),
        'val': transforms.Compose([   
            transforms.Resize((img_height,img_width)),      
            transforms.ToTensor(),
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))            
        ]),        
        }
    elif mode.lower() == "weak":
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_height,img_width)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation([-5,5]),
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))        
        ]),
        'val': transforms.Compose([   
            transforms.Resize((img_height,img_width)),      
            transforms.ToTensor(),
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))            
        ]),
        }
    elif mode.lower() == "hard":
        cutout = Cutout(holes = 1, length = 32)
        data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_height,img_width)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation([-5,5]),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            cutout,
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))        
        ]),
        'val': transforms.Compose([   
            transforms.Resize((img_height,img_width)),      
            transforms.ToTensor(),
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))            
        ]),
        }
    return data_transforms

if __name__ == "__main__":
    data_transforms = get_Transforms(224, 224, mode = 'weak')
    train_valid_dataset = torchvision.datasets.ImageFolder('./retina1_trainvalid/trainvalid', transform=data_transforms['train'])    
    print(train_valid_dataset[0])
