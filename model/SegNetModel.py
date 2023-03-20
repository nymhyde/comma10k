# ----------------------
#  | Model Definition |
# ----------------------


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


import random
from pathlib import Path
import argparse

import cv2, torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import _logger as log

from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

import albumentations as A

pl.seed_everything(1997)



# ** helper functions **

def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')

def pad_to_multiple(x, k=32):
    return int(k*(np.ceil(x/k)))

def get_train_transforms(height=437, width=582):
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        A.PadIfNeeded(pad_to_multiple(height),
                      pad_to_multiple(width),
                      border_mode=cv2.BORDER_CONSTANT,
                      value=0,
                      mask_value=0)
                     ], p=1.0)

def get_valid_transforms(height=437, width=582):
    return A.Compose([
        A.Resize(height=height, width=width, p=1.0),
        A.PadIfNeeded(pad_to_multiple(height),
                      pad_to_multiple(width),
                      border_mode=cv2.BORDER_CONSTANT,
                      value=0,
                      mask_value=0)
                      ], p=1.0)

def get_preprocessing(preprocessing_fn):
    return A.Compose([
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
                    ])



# ** dataset loader class **
class CommaLoader(Dataset):

    def __init__(self, data_path, images_path, preprocess_fn, transforms, class_values):
        super().__init__()

        self.data_path = data_path
        self.images_path = images_path
        self.transforms = transforms
        self.preprocess = get_preprocessing(preprocess_fn)
        self.class_values = class_values
        self.images_folder = 'imgs'
        self.masks_folder = 'masks'


    # default __getitem__ method
    # -- returns x, y

    def __getitem__(self, idx):
        image = self.images_path[idx]
        img = cv2.imread(str(self.data_path/self.images_folder/image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.data_path/self.masks_folder/image), 0).astype('uint8')
        if self.transforms:
            sample = self.transforms(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']

        mask = np.stack([(mask == v) for v in self.class_values], axis=-1).astype('uint8')


        if self.preprocess:
            sample = self.preprocess(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']


        return img, mask


    # default __len__ method
    # -- returns the length of the dataset

    def __len__(self):
        return len(self.images_path)




'''
# -----------------
#  | Model Class |
# -----------------

__init__            :: set up params for Trainer file
model               :: build model Unet
forward             :: forward pass; return logits
loss                :: define loss function
training_step       :: Forward pass -> loss (training set)
validation_setp     :: Forward pass -> loss (validation set)
configure_optim     :: configure optimizers
check data          :: check whether or not we have the files
setup               :: setup dataset for the dataloader
dataloaders         :: train / validation loaders
'''


class SegNet(pl.LightningModule):

    def __init__(self,
                 data_path = '/home/patel4db/comma10k/data/',
                 backbone = 'efficientnet-b0',
                 batch_size = 16,
                 lr = 1e-4,
                 eps = 1e-7,
                 height = 14*32,
                 width = 18*32,
                 num_workers = 40,
                 epochs = 30,
                 gpus = 1,
                 weight_decay = 1e-3,
                 class_values=[41, 76, 90, 124, 161, 0], **kwargs):

        super().__init__()
        self.data_path = Path(data_path)
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.height = height
        self.width = width
        self.num_workers = num_workers
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.eps = eps
        self.class_values = class_values

        self.save_hyperparameters()
        self.preprocess_fn = smp.encoders.get_preprocessing_fn(self.backbone, pretrained='imagenet')

        self.__build_model()


    def __build_model(self):
        '''
        Defining model layers and loss function
        '''

        self.net = smp.Unet(self.backbone, classes=len(self.class_values),
                            activation=None, encoder_weights='imagenet')

        self.loss_func = lambda x,y : torch.nn.CrossEntropyLoss()(x, torch.argmax(y, axis=1))



    def forward(self, x):
        '''
        Forward Pass
        '''

        return self.net(x)


    def loss(self, logits, labels):
        '''
        use the loss function
        '''

        return self.loss_func(logits, labels)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)

        # compute loss and accuracy
        train_loss = self.loss(y_logits, y)
        result = pl.TrainResult(train_loss)

        return result


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)

        # compute loss and accuracy
        val_loss = self.loss(y_logits, y)
        result = pl.EvalResult(checkpoing_on=val_loss)

        return result


    def configure_optimizers(self):

        optimizer = torch.optim.Adam
        optimizer_kwargs = {'eps' : self.eps}

        optimizer = optimizer(self.parameters(),
                              lr = self.lr,
                              weight_decay = self.weight_decay,
                              **optimizer_kwargs)

        return [optimizer]


    def check_data(self):
        assert (self.data_path/'imgs').is_dir(), "Images folder not found"
        assert (self.data_path/'masks').is_dir(), "Masks folder not found"
        assert (self.data_path/'files_trainable').exists(), "trainable fild not found"

        print('Data Tree is found and ready for setup')


    def setup(self, stage):
        images_path = np.loadtxt(self.data_path/'files_trainable', dtype='str').tolist()
        random.shuffle(images_path)

        '''
        For now we are only validating on images ending with * 9.png *
        '''

        self.train_dataset = CommaLoader(data_path = self.data_path,
                                         images_path = [x.split('masks/')[-1] for x in images_path if not x.endswith('9.png')],
                                         preprocess_fn = self.preprocess_fn,
                                         transforms = get_train_transforms(self.height, self.width),
                                         class_values = self.class_values
                                         )

        self.valid_dataset = CommaLoader(data_path=self.data_path,
                                         images_path=[x.split('masks/')[-1] for x in images_path if x.endswith('9.png')],
                                         preprocess_fn=self.preprocess_fn,
                                         transforms=get_valid_transforms(self.height, self.width),
                                         class_values=self.class_values
                                        )


    def __dataloader(self, train):
        '''
        Train and Validation Dataloaders
        '''

        _dataset = self.train_dataset if train else self.valid_dataset
        loader = DataLoader(dataset = _dataset,
                            batch_size = self.batch_size,
                            num_workers = self.num_workers,
                            shuffle = True if train else False)

        return loader


    def train_dataloader(self):
        log.info('Training data loaded')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loaded')
        return self.__dataloader(train=False)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents = [parent_parser])

        parser.add_argument('--backbone',
                        default='efficientnet-b0',
                        type=str,
                        metavar='BK',
                        help='Name as in segmentation_models_pytorch')

        parser.add_argument('--data-path',
                        default='/home/patel4db/comma10k-exp/data/',
                        type=str,
                        metavar='DP',
                        help='data_path')

        parser.add_argument('--epochs',
                        default=30,
                        type=int,
                        metavar='N',
                        help='total number of epochs')

        parser.add_argument('--batch-size',
                        default=32,
                        type=int,
                        metavar='B',
                        help='batch size',
                        dest='batch_size')

        parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='number of gpus to use')

        parser.add_argument('--lr',
                        '--learning-rate',
                        default=1e-4,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='LR')

        parser.add_argument('--eps',
                        default=1e-7,
                        type=float,
                        help='eps for adaptive optimizers',
                        dest='eps')

        parser.add_argument('--height',
                        default=14*32,
                        type=int,
                        help='image height')

        parser.add_argument('--width',
                        default=18*32,
                        type=int,
                        help='image width')

        parser.add_argument('--num-workers',
                        default=40,
                        type=int,
                        metavar='W',
                        help='number of CPU workers',
                        dest='num_workers')

        parser.add_argument('--weight-decay',
                        default=1e-3,
                        type=float,
                        metavar='WD',
                        help='Optimizer weight decay')

        return parser

