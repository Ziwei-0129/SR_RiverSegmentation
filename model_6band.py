"""
The S2S2Net model architecture and data loading modules.

Code structure adapted from Pytorch Lightning project seed at
https://github.com/PyTorchLightning/deep-learning-project-template
"""
import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from Network import S2S2Net
from DataModule import S2S2DataModule




def cli_main(args):
    
    # Set a seed to control for randomness
    pl.seed_everything(seed=args.seed)

    # Load Data
    datamodule: pl.LightningDataModule = \
        S2S2DataModule(
                stage="fit",
                data_root=args.train_data_path,
                batch_size=args.batch_size
        )

    # Initialize Model
    model: pl.LightningModule = S2S2Net(args)


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'pytorch_total_params: {pytorch_total_params}')


    if args.checkpoint_dir is not None:
        checkpoint = os.path.join(args.checkpoint_dir)
        model = model.load_from_checkpoint(checkpoint_path=checkpoint).eval()


    # Setup Tensorboard logger
    tensorboard_logger: pl.loggers.LightningLoggerBase = pl.loggers.TensorBoardLogger(
        save_dir=args.save_dir, name=args.model_type
    )


    # Training:
    trainer: pl.Trainer = pl.Trainer(
        # deterministic=True,
        gpus=2,
        logger=tensorboard_logger,
        max_epochs=args.num_epoch,
        precision=16,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    trainer.fit(model=model, datamodule=datamodule)

    print("\nDone!")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="Path to the pretrained model checkpoint")
    parser.add_argument('--save_dir', type=str, default='tb_logs', help="Path to save the checkpoints")
    parser.add_argument('--train_data_path', type=str, help="Path to the training dataset")
    parser.add_argument('--model_type', type=int, help="dice, bce, or dice_noSR")
    parser.add_argument('--num_epoch', type=int, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--learning_rate', type=float, default=0.00006, help="Training learning rate")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    args = parser.parse_args()


    cli_main(args)
