# DLSP20-Project

### Our Models:
1. Road Map
- Baseline Unet (leaderboard round 2 submission): ```from model_lane_unet import Multi_UNet```
- ResNet-Encoder-Decoder for 6 BEV views (learderboard round 3 submission): ```from model_lane import Multi_Classfier```
- ResNet-Encoder-Decoder for stitched BEV (final report best model): ```from model_lane_res_stitch import Stitch_Classfier```

2. Bounding Box
- Yo2o (leaderboard round 3 submission): ```from model_bboxes import Yo2o``` 
- Stitched Yo4o (final report best model): ```from model_bb_stitch import Yo4o_stitch``` 
- For bounding box predictions and YOLO loss function, see ```loss_bboxes.py```

### Training Notebooks:
1. Road Map: ```Roadmap_Train_ResNet_Enc_Dec.ipynb```
2. Bounding Box ```train_bboxes.py``` & ```fine_tune_bboxes.py```
3. Pretext: ```pretext_image_semantic_inpainting.ipynb```

### Other folders:
- data_exploration: initial data analysis and Bird's Eye View projection experiments
- data_utils: for train/validation data split and sample statistics

Our best model files are in: https://drive.google.com/open?id=1yPUaMaL6IsPqmSGBrziksFYXL0sKkopL

Our final round submission files are in: https://drive.google.com/open?id=1Dcau5I6y6aJoIi5ZD7_0nuqvBhxzrCz_

![Key figures of our model architecture and results](https://github.com/cbs488/DLSP20-Project/blob/master/Highlights.JPG) 
