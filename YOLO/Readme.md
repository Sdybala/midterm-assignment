environment: windows10, python3.8.2, keras2.2.4, use plaidml as backend(Use tensorflow as backend may raise an Error: Using a 'tf.Tensor' as a Pyhton 'bool' is not allowed). 5800X+16G+6800XT


dataset: 
VOC2007, reshaped in shape (448,448,3) and (3,224,224). For yolo we use channel last (448,448,3); for vgg+yolo we use channel first (3,224,224).
Download voc2007 from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar and http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
In fold VOCtest_06-Nov-2007\VOCdevkit: 
Run voc448 to reshape it and put them into subfile 448, run merge_448.py to merge them into one for yolo. Merging is to reduce the time cost of IO 
Run voc224 to reshape it and put them into subfile 224. Run merge_224.py to merge it into one for vgg+yolo. These two are test dataset.
Run voc2yolo.py to get the label in subfold VOC2007yolo, run merge_label to merge the label for the test dataset.
the step of the trainval is the same in VOCtrainval_06-Nov-2007\VOCdevkit.
move these .npy files we get to the same fold of yolov1.py and vgg+yolo.py.

ImageNet 1000 (mini), downloaded from https://www.kaggle.com/ifigotin/imagenetmini-1000, reshaped in shape (224,224,3).
run merge_ImageNet.py in fold archive/imnet to reshape and permutate and merge all the Image data into one. move these four .npy file to the same fold of pretrain.py


yolo:
first run pretrain.py to get the weight of the first 20 conv
Then run yolov1.py to train the yolo network


vgg+yolo:
downloads the vgg16 weight from https://www.kaggle.com/keras/vgg16, decompress it and place it in the same fold.
run vgg+yolo.py to train the vgg+yolo model.
if a Error raises saying module 'keras.backend'  no attribute 'set_image_dim_ordering', add the code below to the __init__.py of the keras.backend (Anaconda3/Lib/site-Packages/keras/backend/__init__.py):
from .common import set_image_dim_ordering
