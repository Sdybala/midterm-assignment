How to use:

1.install tensorflow, preferably GPU version. The python version I use is Python 3.5.

2.Checkout this branch

3.Install python packages (cython, python-opencv, easydict) by running
pip install -r requirements.txt

4.Go to ./data/coco/PythonAPI
Run "python setup.py build_ext --inplace"
Run "python setup.py build_ext install"
Go to ./lib/utils and run "python setup.py build_ext --inplace"

5.Download PyCoco database. The final structure has to look like
data\VOCDevkit2007\VOC2007

6.Download pre-trained VGG16 and place it as data\imagenet_weights\vgg16.ckpt.

7.Run train.py to get the model. Put the model in \output\vgg16\voc_2007_trainval\default.

8.Run demo.py to get the detection result, PR figure and AP data.

If you have any question, please contact me by my email: 731873975@qq.com.
