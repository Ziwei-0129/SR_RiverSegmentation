import typing
import pytorch_lightning as pl
import torch
from Dataset import S2S2Dataset


class S2S2DataModule(pl.LightningDataModule):
    """
    Data preparation code to load Sentinel2 image data and the high resolution
    image mask into a Pytorch DataLoader module (a fancy for-loop generator).

    """

    def __init__(self, stage, data_root, batch_size):

        super().__init__()

        self.stage: str = stage
        self.data_root: str = data_root
        self.batch_size: int = batch_size


    def setup(self, stage: typing.Optional[str] = None) -> torch.utils.data.Dataset:
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        if stage == "fit" or stage is None:  # Training/Validation on chips
            # Combine Sentinel2 and Worldview datasets into one!
            self.dataset: torch.utils.data.Dataset = S2S2Dataset(
                root=self.data_root, train=True  # all river
            )

            # Training/Validation split (80%/20%)
            train_length: int = int(len(self.dataset) * 0.8)
            val_length: int = len(self.dataset) - train_length
            self.dataset_train, self.dataset_val = torch.utils.data.random_split(
                dataset=self.dataset, lengths=(train_length, val_length)
            )

        elif stage == "predict":  # Inference on actual images
            self.dataset: torch.utils.data.Dataset = S2S2Dataset(
                root=self.data_root, train=False
            )

        return self.dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        Set the training batch size here too.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train, batch_size=self.batch_size, num_workers=4
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the validation loop.
        Set the validation batch size here too.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset_val, batch_size=self.batch_size, num_workers=4
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the prediction loop.
        Set the prediction batch size here too.
        """
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1)

