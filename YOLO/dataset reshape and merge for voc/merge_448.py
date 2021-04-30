import os
import numpy as np
filelist=os.listdir("448")

temp1=[]
for i in range(len(filelist)):
    print(i)
    temp=np.load("448/"+filelist[i]).transpose(1,2,0)
    temp1.append(temp)

temp1=np.array(temp1,dtype="uint8")

print("save")

print(temp1.shape)



np.save("pic448",temp1)
np.save("partial_pic448",temp2)
'''


temp1=np.load("pic448.npy")
temp2=np.load("448/"+filelist[15])


for i in range(len(filelist)):
    temp2=np.load("448/"+filelist[i]).transpose(1,2,0)
    temp3=temp1[i]
    if np.sum(temp2==temp3)!=448*448*3:
        print("false",i)

'''

