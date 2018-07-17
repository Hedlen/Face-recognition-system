import torch.utils.data as data
from PIL import Image, ImageFile
import os
import random
ImageFile.LOAD_TRUNCATED_IAMGES = True
class RandomCenterCropAugment(object):
    """docstring for RandomCenterCropAugment"""
    def __init__(self, scale_arg=0.02, trans_arg=0.02, crop_size=110, final_size=128, crop_center_y_offset=25):
        self.scale_arg = scale_arg
        self.trans_arg = trans_arg
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
    def __call__(self, image):
        scale_factor = (random.randint(1000,1000)*1.0/500.0 - 1)*self.scale_arg;
        centerx_factor = (random.randint(1000,1000)*1.0/500.0 - 1)*self.trans_arg;
        centery_factor = (random.randint(1000,1000)*1.0/500.0 - 1)*self.trans_arg;
        width,height=image.size
        crop_size_aug = self.crop_size*(1+scale_factor); #110
        center_x=width/2.*(1+centerx_factor)
        center_y=(height/2. + self.crop_center_y_offset)*(1+centery_factor)
        if(center_x < crop_size_aug/2):
            crop_size_aug = center_x*2-0.5
        if(center_y < crop_size_aug/2):
            crop_size_aug = center_y*2-0.5
        if(center_x + crop_size_aug/2 >= width):
            crop_size_aug = (width-center_x)*2 - 0.5
        if(center_y + crop_size_aug/2 >= height):
            crop_size_aug = (height-center_y)*2 - 0.5
        side=crop_size_aug/2
        rect = (int(center_x-side), int(center_y-side), int(center_x+side), int(center_y+side))
        cropped = image.crop(rect)
        cropped = cropped.resize((self.final_size,self.final_size))
        return cropped

def PIL_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
    else:
        return img


def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList
def default_reader_2(filedict):
    imgList = []
    for root,listfile in filedict.items():
            with open(listfile,'r') as f:
                lines = f.readlines()
                for line in lines:
                    imgPath, label = line.strip().split(' ') 
            #imgPath, label = line.strip().split(' ')
                    imgList.append((root+'/'+imgPath, int(label)))
    return imgList

class ImageList(data.Dataset):
    '''
     Args:
        root (string): Root directory path.
        fileList (string): Image list file path
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    '''

    def __init__(self,filedict, transform=None, list_reader=default_reader_2, loader=PIL_loader):
        #self.root = root
        self.imgList = list_reader(filedict)
        self.transform = transform
        self.loader = loader
        self.crop = RandomCenterCropAugment() 
    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        #img = self.loader(os.path.join(self.root, imgPath))
        #img = self.crop(img)
        img = self.loader(imgPath)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)
