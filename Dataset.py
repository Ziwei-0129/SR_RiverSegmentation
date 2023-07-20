import glob
import os
import argparse
import typing

import numpy as np
import rioxarray
import torch
import torchgeo.datasets


class S2S2Dataset(torchgeo.datasets.VisionDataset):
    """
    Training dataset for the Sentinel-2 Super Resolution Segmentation model.

    There are 3 image triples:
    1. image - Sentinel-2 RGB-NIR image at 10m resolution (4, 512, 512)
    2. mask - Binary segmentation mask at 2m resolution (1, 2560, 2560)
    3. hres - High resolution RGB-NIR image at 2m resolution (4, 2560, 2560)
    """

    def __init__(
        self,
        root: str = "",  # Train/Validation chips
        train: bool = True,  # Whether to load training set or predict only
        transforms: typing.Optional[
            typing.Callable[
                [typing.Dict[str, torch.Tensor]], typing.Dict[str, torch.Tensor]
            ]
        ] = None,
    ):
        self.root: str = root
        self.train: bool = train
        self.transforms = transforms

        img_path: str = (
            os.path.join(self.root, "image") if self.train else os.path.join(self.root)
        )
        self.ids: list = [int(id) for id, _ in enumerate(os.listdir(path=img_path))]

    def __getitem__(self, index: int = 0) -> typing.Dict[str, torch.Tensor]:
        """
        Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        if self.train:
            image: torch.Tensor = torch.from_numpy(
                np.load(
                    file=os.path.join(self.root, "image", f"SEN2_{index:04d}.npy")
                ).astype(np.int16)
            )
            mask: torch.Tensor = torch.from_numpy(
                np.load(file=os.path.join(self.root, "mask", f"MASK_{index:04d}.npy"))
            )
            hres: torch.Tensor = torch.from_numpy(
                np.load(
                    file=os.path.join(self.root, "hres", f"HRES_{index:04d}.npy")
                ).astype(np.int16)
            )

            sample: dict = {"image": image, "mask": mask, "hres": hres}

        else:
            filename: str = glob.glob(
                os.path.join(self.root, f"{index:04d}", "S2*.tif")
            )[0]
            with rioxarray.open_rasterio(filename=filename) as rds:
                assert rds.ndim == 3  # Channel, Height, Width
                assert rds.shape[0] == 6  # 6 bands/channels (RGB+NIR)
                sample: dict = {
                    "image": torch.as_tensor(data=rds.data.astype(np.int16))
                }

        if self.transforms is not None:
            sample: typing.Dict[str, torch.Tensor] = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """
        Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

