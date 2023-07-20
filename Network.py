import typing
import mmseg.models
import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score



class S2S2Net(pl.LightningModule):
    """
    Neural network for performing Super-Resolution Semantic Segmentation on
    Sentinel 2 optical satellite imagery.

    Implemented using Pytorch Lightning.
    """

    def __init__(self, args):
        """
        Define layers of the Segmentation Network.
        """
        super().__init__()

        self.model_type = args.model_type
        self.lr = args.learning_rate


        ## Input Module (Encoder/Backbone).
        self.deeplabv3plus_backbone = mmseg.models.backbones.ResNetV1c(
            in_channels=6,  # RGB+NIR
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            # norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
        )

        ## Middle Module (Decoder).
        # This head is the implementation of DeepLabV3+.
        self.deeplabv3plus_head = mmseg.models.decode_heads.DepthwiseSeparableASPPHead(
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            c1_in_channels=256,
            c1_channels=48,
            dropout_ratio=0.1,
            num_classes=16,
            # norm_cfg=norm_cfg,
            align_corners=False,
        )

        if self.model_type == "dice":
            ## Upsampling layers (Output).
            # 1st and 2nd layers are to get back original image size
            # 3rd and 4th layers are to get 5x super-resolution result
            # Each of the upsampling layers are followed by a Conv2D layer
            self.upsample_0 = torch.nn.Upsample(scale_factor=2, mode="bicubic")
            self.post_upsample_conv_layer_0 = torch.nn.Conv2d(
                in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1
            )
            self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode="bicubic")
            self.post_upsample_conv_layer_1 = torch.nn.Conv2d(
                in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1
            )

            self.upsample_2 = torch.nn.Upsample(scale_factor=2, mode="bicubic")
            self.post_upsample_conv_layer_2 = torch.nn.Conv2d(
                in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1
            )
            self.upsample_3 = torch.nn.Upsample(scale_factor=2.5, mode="bicubic")
            self.post_upsample_conv_layer_3 = torch.nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1
            )

        elif self.model_type == "bce":
            ## Upsampling layers (Output).
            # 1st and 2nd layers are to get back original image size
            # 3rd and 4th layers are to get 5x super-resolution result
            # Each of the upsampling layers are followed by a Conv2D layer
            self.upsample_0 = torch.nn.Upsample(scale_factor=2, mode="bicubic")
            self.post_upsample_conv_layer_0 = torch.nn.Conv2d(
                in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1
            )
            self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode="bicubic")
            self.post_upsample_conv_layer_1 = torch.nn.Conv2d(
                in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1
            )

            self.upsample_2 = torch.nn.Upsample(scale_factor=2, mode="bicubic")
            self.post_upsample_conv_layer_2 = torch.nn.Conv2d(
                in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1
            )
            self.upsample_3 = torch.nn.Upsample(scale_factor=2.5, mode="bicubic")
            self.post_upsample_conv_layer_3 = torch.nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1
            )

        elif self.model_type == "dice_noSR":
            ## Upsampling layers (Output).
            self.upsample = torch.nn.Upsample(scale_factor=4, mode="bilinear")

        else:
            print("Unknown network type...")
            exit(1)

        # Evaluation metrics to know how good the segmentation results are
        self.iou = torchmetrics.JaccardIndex(num_classes=2)
        self.f1_score = torchmetrics.F1Score(num_classes=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Inference/Prediction).
        """
        output_tensors: typing.List[torch.Tensor] = self.deeplabv3plus_backbone(x)
        assert len(output_tensors) == 4

        depplabv3plus_output: torch.Tensor = self.deeplabv3plus_head(output_tensors)

        ## Step 3. Do a series of bilinear interpolation upsampling + Conv2d
        up0_output: torch.Tensor = (self.upsample_0(depplabv3plus_output))
        up0_conv_output: torch.Tensor = self.post_upsample_conv_layer_0(up0_output)

        up1_output: torch.Tensor = (self.upsample_1(up0_conv_output))
        up1_conv_output: torch.Tensor = self.post_upsample_conv_layer_1(up1_output)

        up2_output: torch.Tensor = (self.upsample_2(up1_conv_output))
        up2_conv_output: torch.Tensor = self.post_upsample_conv_layer_2(up2_output)

        up3_output: torch.Tensor = (self.upsample_3(up2_conv_output))
        up3_conv_output: torch.Tensor = self.post_upsample_conv_layer_3(up3_output)

        return up3_conv_output

    def evaluate(
            self, batch: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        # Compute the loss for a single batch in the training or validation step.

        x: torch.Tensor = batch["image"].float()  # Input Sentinel-2 image
        y: torch.Tensor = batch["mask"]  # Groundtruth binary mask
        y_highres: torch.Tensor = batch["hres"]  # High resolution image
        # y = torch.randn(8, 1, 2560, 2560)
        # y_highres = torch.randn(8, 4, 2560, 2560)

        y_hat: typing.Dict[str, torch.Tensor] = self(x)

        # 2: Semantic Segmentation loss (Sigmoid + Cross-entropy Loss)
        if self.model_type == "dice":
            loss_fn = nn.BCEWithLogitsLoss()
        elif self.model_type == "bce":
            loss_fn = nn.BCEWithLogitsLoss()
        elif self.model_type == "dice_noSR":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            print("Unknown network type...")
            exit(1)

        segmmask_loss: torch.Tensor = loss_fn(y_hat, y)

        losses: typing.Dict[str, torch.Tensor] = {
            "loss_segmmask": segmmask_loss.detach().item(),
        }

        # Calculate metrics to determine how good results are
        iou_score: torch.Tensor = self.iou(  # Intersection over Union
            preds=y_hat.squeeze(),
            target=(y > 0.5).squeeze().to(dtype=torch.int8),  # binarize
        )
        f1_score: torch.Tensor = self.f1_score(  # F1 Score
            preds=y_hat.ravel(),
            target=(y > 0.5).ravel().to(dtype=torch.int8),  # binarize
        )

        _precision = precision_score(
            y_true=((y.cpu().detach()) > 0.5).to(dtype=torch.int8).ravel(),
            y_pred=((y_hat.cpu().detach()) > 0.5).to(dtype=torch.int8).ravel(),
            average='binary',
            zero_division=1
        )

        _recall = recall_score(
            y_true=((y.cpu().detach()) > 0.5).to(dtype=torch.int8).ravel(),
            y_pred=((y_hat.cpu().detach()) > 0.5).to(dtype=torch.int8).ravel(),
            average='binary',
            zero_division=1
        )

        metrics: typing.Dict[str, torch.Tensor] = \
            {"iou": iou_score, "f1": f1_score, "recall": _recall, "precision": _precision}

        total_loss: torch.Tensor = segmmask_loss
        return {"loss": total_loss, **losses, **metrics}

    def training_step(
            self, batch: typing.Dict[str, torch.Tensor], batch_idx: int
    ) -> dict:
        """
        Logic for the neural network's training loop.
        """

        losses_and_metrics: dict = self.evaluate(batch=batch)
        self.log_dict(dictionary=losses_and_metrics, prog_bar=True)

        # Log training loss and metrics to Tensorboard
        if self.logger is not None and hasattr(self.logger.experiment, "add_scalars"):
            for metric_name, metric_value in losses_and_metrics.items():
                self.logger.experiment.add_scalars(
                    main_tag=metric_name,
                    tag_scalar_dict={"train": metric_value},
                    global_step=self.global_step,
                    # epoch=self.current_epoch,
                )

        return losses_and_metrics["loss"]

    def validation_step(
            self,
            batch: typing.Tuple[typing.List[torch.Tensor], typing.List[typing.Dict]],
            batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's validation loop.
        """
        val_losses_and_metrics: dict = self.evaluate(batch=batch)

        self.log_dict(
            dictionary={
                f"val_{key}": value for key, value in val_losses_and_metrics.items()
            },
            prog_bar=True,
        )

        # Log validation loss and metrics to Tensorboard
        if self.logger is not None and hasattr(self.logger.experiment, "add_scalars"):
            for metric_name, metric_value in val_losses_and_metrics.items():
                self.logger.experiment.add_scalars(
                    main_tag=metric_name,
                    tag_scalar_dict={"validation": metric_value},
                    global_step=self.global_step,
                    # epoch=self.current_epoch,
                )

        return val_losses_and_metrics["loss"]

    def predict_step(
            self,
            batch: typing.Dict[str, torch.Tensor],
            batch_idx: int,
            dataloader_idx: typing.Optional[int] = None,
    ):
        """
        Logic for the neural network's prediction loop.
        """
        x: torch.Tensor = batch["image"].float()  # Input Sentinel-2 image

        y_hat: torch.Tensor = self(x)

        return torch.sigmoid(input=y_hat)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(), lr=self.lr, weight_decay=0.01
        )

