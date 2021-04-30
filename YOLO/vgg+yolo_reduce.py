import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras.models as models
import keras.layers as layers
import keras.backend as K

K.set_image_dim_ordering('th')
import keras.initializers as initializers
import keras.regularizers as regularizers
import numpy as np
import keras.callbacks as callbacks

xInput=layers.Input(shape=(3,224,224,))

x=layers.Conv2D(filters=64,kernel_size=(3,3),strides=1, padding='same', kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name='block1_conv1', trainable=False)(xInput)
x=layers.Conv2D(filters=64,kernel_size=(3,3),strides=1, padding='same', kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name='block1_conv2', trainable=False)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)


x=layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block2_conv1", trainable=False)(x)
x=layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block2_conv2", trainable=False)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)


x=layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block3_conv1", trainable=False)(x)
x=layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block3_conv2", trainable=False)(x)
x=layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block3_conv3", trainable=False)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block4_conv1", trainable=False)(x)
x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block4_conv2", trainable=False)(x)
x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block4_conv3", trainable=False)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)

x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block5_conv1", trainable=False)(x)
x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block5_conv2", trainable=False)(x)
x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation="relu",use_bias=True,name="block5_conv3", trainable=False)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)


x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True)(x)
x = layers.BatchNormalization(axis=3)(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True)(x)
x = layers.BatchNormalization(axis=3)(x)
x=layers.LeakyReLU(alpha=0.1)(x)


# x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True)(x)
# x = layers.BatchNormalization(axis=3)(x)
# x=layers.LeakyReLU(alpha=0.1)(x)
# x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True)(x)
# x = layers.BatchNormalization(axis=3)(x)
# x=layers.LeakyReLU(alpha=0.1)(x)


x=layers.Flatten()(x)
x=layers.Dense(4096,use_bias=True,activation=None)(x)

x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.Dense(1470,use_bias=True,activation="linear")(x)


x=layers.Reshape((7,7,30))(x)

model=models.Model(input=xInput,output=x)
model.summary()





#loss
def classloss(y_true,y_pred):
    trueprob=y_true[...,10:]
    predprob=y_pred[...,10:]
    err=(trueprob-predprob)
    ident=y_true[...,:1]
    delta=ident*err
    loss=K.sum(delta*delta)
    return loss

def xyloss(y_true,y_pred):
    ident1=y_true[...,:1]
    ident2=y_true[...,5:6]
    xytrue1 = y_true[..., 1:3]
    xytrue2 = y_true[..., 6:8]
    xypred1 = y_pred[..., 1:3]
    xypred2 = y_pred[..., 6:8]
    err1=xytrue1-xypred1
    err2=xytrue2-xypred2

    delta1 = ident1*err1
    delta2 = ident2*err2
    loss = K.sum(delta1*delta1)+K.sum(delta2*delta2)
    return loss
def whloss(y_true,y_pred):
    ident1=y_true[...,:1]
    ident2=y_true[...,5:6]
    xytrue1 = K.sqrt(y_true[..., 3:5])
    xytrue2 = K.sqrt(y_true[..., 8:10])
    xypred1 = K.sqrt(y_pred[..., 3:5])
    xypred2 = K.sqrt(y_pred[..., 8:10])
    err1=xytrue1-xypred1
    err2=xytrue2-xypred2
    delta1 = ident1*err1
    delta2 = ident2*err2
    loss = K.sum(delta1*delta1)+K.sum(delta2*delta2)
    return loss

def IOUloss(y_true,y_pred,lamnoobj=2):
    conf1=y_pred[...,:1]*IOU(y_true[...,:5],y_pred[...,:5])
    conf2=y_pred[...,5:6]*IOU(y_true[...,5:10],y_pred[...,5:10])
    ident1=y_true[...,:1]
    ident2=y_true[...,5:6]
    lossobj=K.sum(ident1*(conf1-ident1)*(conf1-ident1))+K.sum(ident2*(conf2-ident2)*(conf2-ident2))
    losstotal=K.sum((conf1-ident1)*(conf1-ident1))+K.sum((conf2-ident2)*(conf2-ident2))
    loss=lamnoobj*losstotal+(1-lamnoobj)*lossobj
    return loss

def IOU(box1,box2):
    xlow1=box1[...,1:2]-0.5*box1[...,3:4]*7.0
    xhigh1=box1[...,1:2]+0.5*box1[...,3:4]*7.0
    ylow1=box1[...,2:3]-0.5*box1[...,4:5]*7.0
    yhigh1=box1[...,2:3]+0.5*box1[...,4:5]*7.0
    xlow2=box2[...,1:2]-0.5*box2[...,3:4]*7.0
    xhigh2=box2[...,1:2]+0.5*box2[...,3:4]*7.0
    ylow2=box2[...,2:3]-0.5*box2[...,4:5]*7.0
    yhigh2=box2[...,2:3]+0.5*box2[...,4:5]*7.0
    xinterlow=np.maximum(xlow1,xlow2)
    xinterhigh=np.minimum(xhigh1,xhigh2)
    yinterlow=np.maximum(ylow1,ylow2)
    yinterhigh=np.minimum(yhigh1,yhigh2)
    zeros=0.0*xinterlow
    inter=K.maximum(zeros,xinterhigh-xinterlow)*K.maximum(zeros,yinterhigh-yinterlow)
    union=box1[...,3:4]*7.0*box1[...,4:5]*7.0+box2[...,3:4]*7.0*box2[...,4:5]*7.0-inter
    return inter/union

def totalloss(y_true,y_pred,lamcoord=5,lamnoobj=0.5):
    loss=lamcoord*classloss(y_true,y_pred)+xyloss(y_true,y_pred)+whloss(y_true,y_pred)+IOUloss(y_true,y_pred,lamnoobj)
    return loss

#Adam,Adadelta,Adagrad,Adanax,Nadam,


testdata=np.load("pic224cf.npy")[200:]
testlabel=np.load("label.npy")[200:]
print(testdata.shape)
print(testlabel.shape)

model.compile(loss=totalloss,optimizer="Adam")
model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",by_name=True)
callback=callbacks.EarlyStopping(min_delta=1,patience=15,restore_best_weights=True)
history=model.fit(testdata,testlabel,epochs=120,batch_size=64,validation_split=0.2,callbacks=[callback])
model.save_weights("vggyolo.h5")

from matplotlib import pyplot as plt

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
