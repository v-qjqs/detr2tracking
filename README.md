# detr2tracking
DETR based ([End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)) Multiple Object Tracking

## Install
For train/evaluate DETR models, mmcv and another common packages are needed.

For generate tracking results and calculate metrics(mota/idf1, etc.) on validation dataset, following the installation of [Neural Solver](https://github.com/dvl-tum/mot_neural_solver)

All the requirements are list in *requirements.txt*

## Running Process
### Datasets
```python
# train
mot17_data=['MOT17-02-FRCNN','MOT17-05-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN','MOT17-13-FRCNN']
mot15_data=['KITTI-17','ETH-Sunnyday','ETH-Bahnhof','PETS09-S2L1','TUD-Stadtmitte']

# validation
mot17_data_eval=['MOT17-11-FRCNN', 'MOT17-04-FRCNN']
```

### Pretrain ReID model
[TorchReID](https://kaiyangzhou.github.io/deep-person-reid/user_guide) is used for training ReID model and generating reid features on the validation set.  *resnet50_fc512* model is used.
### Prepare ReID and Position Encoding Features
```python
# associate the name of ReID feature with its corresponding ground-truth bbox. 
# For validation:
python feats_extract/reid2gtbbox.py # or python reid2gtbbox_traindata.py for training data

# generate position encoding feature and concat ground-truth bbox infomation with it
python feats_extract/pos_encoding.py

# merge ReID feature with concated position encoding feature. The merged feature is used for construct 
# dataset which is further used for DETR train/test.
python feats_extract/reid_posfeat_merge.py
```

### Train
```python
# training of DETR model
cd mot_trainer
scripts/dis_train.sh
```

### Evaluation
```python
# generate matched label infomation (Pandas DataFrame) on validation set based on trained DETR model
scripts/dis_eval.sh

# generate txt file of tracking results and calculate metrics (mota/idf1, etc.)
cd mot_tracker
python tracker.py         
# The above tracker.py does the following roughly for each dataset sequence: (modified from the evaluation process of Neural Solver) 
# 1) construct node info: read from gt.txt and construct pandas.DataFrame, filter invalid gt bbox (visible<0.2 or clss id not in [1,2])
# 2) merge matched/associated info (or called edge info, inferred from trained DETR model) with DataFrame in 1)
# 3) construct graph
# 4) graph project (same to Neural Solver) to constrain the in/out degree <=1 for each node
# 5) connected components calculation, tracking results generation and metrics calculation 
```
