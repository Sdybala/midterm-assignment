import xml.etree.ElementTree as ET
import numpy as np

classes_names = ["aeroplane", "bicycle", "bird", "boat", "bottle",

                 "bus", "car", "cat", "chair", "cow", "diningtable", "dog",

                 "horse", "motorbike", "person", "pottedplant", "sheep",

                 "sofa", "train", "tvmonitor"]
v = 0
morethan3 = []
for img_number in range(1, 9964):
    cxywh_1 = np.zeros([7, 7, 5])
    cxywh_2 = np.zeros([7, 7, 5])
    label = np.zeros(([7, 7, 20]))
    filename = str
    tree = ET.parse('VOC2007/Annotations/' + str(img_number).rjust(6, '0') + '.xml')
    root = tree.getroot()
    for width in root.iter('width'):
        width = int(width.text)
    for height in root.iter('height'):
        height = int(height.text)
    for object in root.iter('object'):
        for bndbox in object.find('bndbox'):
            for xmin in bndbox.iter('xmin'):
                xmin = int(xmin.text)
            for xmax in bndbox.iter('xmax'):
                xmax = int(xmax.text)
            for ymin in bndbox.iter('ymin'):
                ymin = int(ymin.text)
            for ymax in bndbox.iter('ymax'):
                ymax = int(ymax.text)

        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        cxfloat = xmid * 7 / width
        cyfloat = ymid * 7 / height
        cx = int(cxfloat)
        cy = int(cyfloat)
        x = cxfloat - cx
        y = cyfloat - cy
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        if cxywh_1[cx, cy, 0] == 0:
            cxywh_1[cx, cy, 0] = 1
            cxywh_1[cx, cy, 1] = x
            cxywh_1[cx, cy, 2] = y
            cxywh_1[cx, cy, 3] = w
            cxywh_1[cx, cy, 4] = h
            cxywh_2[cx, cy, 0] = 1
            cxywh_2[cx, cy, 1] = x
            cxywh_2[cx, cy, 2] = y
            cxywh_2[cx, cy, 3] = w
            cxywh_2[cx, cy, 4] = h
            label[cx, cy, classes_names.index(object[0].text)] = 1
        else:
            if cxywh_1[cx, cy, 1] == cxywh_2[cx, cy, 1]:
                cxywh_2[cx, cy, 0] = 1
                cxywh_2[cx, cy, 1] = x
                cxywh_2[cx, cy, 2] = y
                cxywh_2[cx, cy, 3] = w
                cxywh_2[cx, cy, 4] = h
                label[cx, cy, classes_names.index(object[0].text)] = 1
            else:
                continue
    np.save("VOC2007yolo/"+str(img_number).rjust(6, '0'), np.concatenate((cxywh_1, cxywh_2, label), 2))
