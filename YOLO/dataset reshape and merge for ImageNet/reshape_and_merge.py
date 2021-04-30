import os

import numpy as np
from PIL import Image

cata=os.listdir("val")
label=[]
datalist=[]
k=224
shape=(k,k)

for i in range(1000):
    print(i)
    namelist=os.listdir("val/"+cata[i])
    labelthis=np.zeros((1000))
    labelthis[i]=1
    for name in namelist:
        im = Image.open('val/' + cata[i] +"/"+name)
        imnew=im.resize((k,k))
        imgdata = np.array(imnew.getdata())
        try:
            r = imgdata[:, 0]
            r = r.reshape(shape, order='F')

            g = imgdata[:, 1]
            g = g.reshape(shape, order='F')

            b = imgdata[:, 2]
            b = b.reshape(shape, order='F')
            data = np.array([r, g, b],dtype='uint8')
        except IndexError:
            data=imgdata.reshape(shape, order='F')
            data=np.array([data,data,data],dtype='uint8')

        data=data.transpose(1,2,0)
        datalist.append(data)
        label.append(labelthis)


data=np.array(datalist,dtype='uint8')
label=np.array(label,dtype='uint8')
print(data.shape)
permutation=list(np.random.permutation(data.shape[0]))
data=data[permutation,:,:,:]
label=label[permutation,:]


np.save("Imagenet_valid",data)
np.save("label_valid",label)

cata=os.listdir("train")
label=[]
datalist=[]
k=224
shape=(k,k)

for i in range(1000):
    print(i)
    namelist=os.listdir("train/"+cata[i])
    labelthis=np.zeros((1000))
    labelthis[i]=1
    for name in namelist:
        im = Image.open('train/' + cata[i] +"/"+name)
        imnew=im.resize((k,k))
        imgdata = np.array(imnew.getdata())
        try:
            r = imgdata[:, 0]
            r = r.reshape(shape, order='F')

            g = imgdata[:, 1]
            g = g.reshape(shape, order='F')

            b = imgdata[:, 2]
            b = b.reshape(shape, order='F')
            data = np.array([r, g, b],dtype='uint8')
        except IndexError:
            data=imgdata.reshape(shape, order='F')
            data=np.array([data,data,data],dtype='uint8')

        data=data.transpose(1,2,0)
        datalist.append(data)
        label.append(labelthis)


data=np.array(datalist,dtype='uint8')
label=np.array(label,dtype='uint8')
print(data.shape)
permutation=list(np.random.permutation(data.shape[0]))
data=data[permutation,:,:,:]
label=label[permutation,:]


np.save("Imagenet_train",data)
np.save("label_train",label)
