import os
import numpy as np
filelist=os.listdir("VOC2007yolo")

temp1=[]

for i in range(len(filelist)):
    print(i)
    temp=np.load("VOC2007yolo/"+filelist[i])
    temp1.append(temp)

temp1=np.array(temp1)
print("save")
np.save("label",temp1)
'''
filelist=os.listdir("VOC2007yolo")


temp1=np.load("label.npy")
temp2=np.load("VOC2007yolo/"+filelist[15])

temp3=temp1[15]
print(np.sum(temp2==temp3))

for i in range(9963):
    temp2=np.load("VOC2007yolo/"+filelist[i])
    temp3=temp1[i]
    if np.sum(temp2==temp3)!=1470:
        print(i)
'''
