# SR_RiverSegmentation
Codes for Super-Resolution Deep Neural Networks for Water Classification 3 from Free Multispectral Satellite Imagery

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
python train_model_6band.py --checkpoint_dir ckpts --data_path TensorFlowRecords --figure_path figs --data_dim 1 --model_index 1 --num_epoch 2 --batch_size 24 --learning_rate 0.1
```


