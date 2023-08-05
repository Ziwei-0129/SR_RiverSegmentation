# SR_RiverSegmentation
Codes for Super-Resolution Deep Neural Networks for Water Classification from Free Multispectral Satellite Imagery

### Requirements
- OpenCV
- rasterio
- xarray 
- rioxarray
- geopandas
- python=3.9.10
- pytorch=1.10.0
- pytorch-lightning=1.5.10

To create the PyTorch environment for training and inference please refer the file [environment.yml](https://github.com/Ziwei-0129/SR_RiverSegmentation/blob/main/environment.yml)

### Running Our Codes

**Model Training:**
```python
python train_model_6band.py --save_dir tb_logs --train_data_path .../SuperResolution/chips/npy_6band_nonan --model_type dice --data_dim 1 --num_epoch 100 --batch_size 32 --learning_rate 0.00001 --seed 42
```

**Format of training dataset**

There are three image triples (you need three data folders under the training dataset path):
1. image - Sentinel-2 RGB-NIR image at 10m resolution (6, 512, 512)
2. mask - Binary segmentation mask at 2m resolution (1, 2560, 2560)
3. hres - High resolution RGB-NIR image at 2m resolution (6, 2560, 2560)


**Inference:**

We provide three designs of the Sentinel-2 Super Resolution Segmentation model:
```
1: Baseline (dice_noSR): DeepLabV3+ with ResNet50 using DiceBCE loss without Super-resolution operations
2: SR-Dice (dice): DeepLabV3+ with ResNet50 using DiceBCE loss
3: SR-BCE (bce): DeepLabV3+ with ResNet50 using BCE loss
```

Our pretrained model checkpoints can be downloaded from [checkpoints](https://drive.google.com/drive/folders/1u3jlJdKWEbR0TaA9opYDEVvcF4YA5W6p?usp=sharing)


### Acknowledgement

The segmentation networks are implemented based on the [OpenMMLab](https://github.com/open-mmlab/mmsegmentation)
