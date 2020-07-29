# detr2tracking
DETR based ([End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)) Multiple Object Tracking

## Install
For train/evaluate DETR models, mmcv and another common packages are needed. *resnet50_fc512* is used.

For generate tracking results and calculate metrics(mota/idf1, etc.) on validation dataset, following the installation of [Neural Solver](https://github.com/dvl-tum/mot_neural_solver)

## Running Process
### Pretrain ReID model
[TorchReID](https://kaiyangzhou.github.io/deep-person-reid/user_guide) is used for training ReID model and generating reid features on the validation set. 
### Prepare ReID and Position Encoding Features
'''
python 
'''

### Train
### Evaluation
