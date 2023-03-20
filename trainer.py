#! /usr/bin/env python3

# -----------------------------------------------------
# Runs a model on a single node with multi-gpu support
# -----------------------------------------------------

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from pathlib import Path
from argparse import ArgumentParser
from model.SegNetModel import *

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import Callback


seed_everything(1997)

def setup_callbacks_loggers(args):

    log_path = Path('/home/patel4db/comma10k/data/')
    name = args.backbone
    version = args.version
    tb_logger = TensorBoardLogger(log_path, name=name, version=version)
    ckpt_callback = ModelCheckpoint(dirpath = Path(tb_logger.log_dir)/'checkpoints/',
                                    filename = '{epoch:.02d}___{val_loss:.4f}',
                                    monitor='val_loss', save_top_k=10, save_last=True)

    return ckpt_callback, tb_logger



def main(args):
    """
    Training Loop
    """


    if args.seed_from_checkpoint:
        print('Model Seeded')
        model = SegNet.load_from_checkpoint(args.seed_from_checkpoint, **vars(args))

    else:
        print('Model from Scratch')
        model = SegNet(**vars(args))


    ckpt_callback, tb_logger = setup_callbacks_loggers(args)

    trainer = Trainer(callbacks=ckpt_callback,
                      logger = tb_logger,
                      gpus = args.gpus,
                      enable_progress_bar = True,
                      enable_model_summary = True,
                      min_epochs = args.epochs,
                      max_epochs = args.epochs,
                      precision = 16,
                      amp_backend='native',
                      accelerator='auto',
                      strategy="ddp",
                      benchmark = True,
                      sync_batchnorm=True,
                      resume_from_checkpoint=args.resume_from_checkpoint)



    trainer.logger.log_hyperparams(model.hparams)

    trainer.fit(model)


def run_from_cli():

    root_dir = os.path.dirname(os.path.realpath(__file__))

    parent_parser = ArgumentParser(add_help=False)
    parser = SegNet.add_model_specific_args(parent_parser)

    parser.add_argument('--version',
                        default = None,
                        type = str,
                        metavar = 'V',
                        help = 'version of the net')

    parser.add_argument('--resume-from-checkpoint',
                        default = None,
                        type = str,
                        metavar = 'RFC',
                        help = 'path to checkpoint')

    parser.add_argument('--seed-from-checkpoint',
                        default = None,
                        type = str,
                        metavar = 'SFC',
                        help = 'ptah to checkpoint seed')

    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    run_from_cli()
