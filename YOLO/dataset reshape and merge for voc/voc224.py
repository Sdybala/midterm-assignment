import numpy as np
from PIL import Image

k=224
shape=(k,k)

for n in range(1, 9964):
    im = Image.open('VOC2007/JPEGImages/' + str(n).rjust(6, '0') + '.jpg')
    im=im.resize((k,k))

    imgdata = np.array(im.getdata())

    r = imgdata[:, 0]
    r = r.reshape(shape, order='F')


    g = imgdata[:, 1]
    g = g.reshape(shape, order='F')


    b = imgdata[:, 2]
    b = b.reshape(shape, order='F')


    data = np.array([r, g, b],dtype='uint8')
    np.save("224/"+str(n).rjust(6, '0'),data)
