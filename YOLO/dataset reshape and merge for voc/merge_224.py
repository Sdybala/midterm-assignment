import os
import numpy as np
filelist=os.listdir("224")

temp1=[]
for i in range(len(filelist)):
    print(i)
    temp=np.load("224/"+filelist[i])
    temp1.append(temp)

temp1=np.array(temp1,dtype="uint8")

print("save")
np.save("pic224",temp1)

'''


temp1=np.load("pic224.npy")
temp2=np.load("224/"+filelist[15])


for i in range(len(filelist)):
    temp2=np.load("224/"+filelist[i]).transpose(1,2,0)
    temp3=temp1[i]
    if np.sum(temp2==temp3)!=224*224*3:
        print("false",i)


'''
