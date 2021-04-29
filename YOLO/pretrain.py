import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras.models as models
import keras.layers as layers
import keras.backend as K
import keras.initializers as initializers
import keras.regularizers as regularizers
import numpy as np
import keras.callbacks as callbacks
import keras.losses as losses

xInput=layers.Input(shape=(224,224,3,))

x=layers.Conv2D(filters=64,kernel_size=(7,7),strides=2, padding='same', kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name='conv1')(xInput)
x = layers.BatchNormalization(axis=3,name="norm1")(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)


x=layers.Conv2D(filters=192,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name='conv2')(x)
x = layers.BatchNormalization(axis=3,name="norm2")(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)


x=layers.Conv2D(filters=128,kernel_size=(1,1),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv3")(x)
x = layers.BatchNormalization(axis=3,name="norm3")(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv4")(x)
x = layers.BatchNormalization(axis=3,name="norm4")(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.Conv2D(filters=256,kernel_size=(1,1),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv5")(x)
x = layers.BatchNormalization(axis=3,name="norm5")(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv6")(x)
x = layers.BatchNormalization(axis=3,name="norm6")(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)


for i in range(4):
    x=layers.Conv2D(filters=256,kernel_size=(1,1),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv"+str(i*2+7))(x)
    x = layers.BatchNormalization(axis=3,name="norm"+str(i*2+7))(x)
    x=layers.LeakyReLU(alpha=0.1)(x)
    x=layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv"+str(i*2+8))(x)
    x = layers.BatchNormalization(axis=3,name="norm"+str(i*2+8))(x)
    x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.Conv2D(filters=512,kernel_size=(1,1),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv15")(x)
x = layers.BatchNormalization(axis=3,name="norm15")(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.Conv2D(filters=1024,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv16")(x)
x = layers.BatchNormalization(axis=3,name="norm16")(x)
x=layers.LeakyReLU(alpha=0.1)(x)
x=layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)


for i in range(2):
    x=layers.Conv2D(filters=512,kernel_size=(1,1),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv"+str(i*2+17))(x)
    x = layers.BatchNormalization(axis=3,name="norm"+str(i*2+17))(x)
    x=layers.LeakyReLU(alpha=0.1)(x)
    x=layers.Conv2D(filters=1024,kernel_size=(3,3),strides=(1, 1), padding='same',kernel_initializer=initializers.truncated_normal(0.0, 0.01), kernel_regularizer=regularizers.l2(5e-4),activation=None,use_bias=True,name="conv"+str(i*2+18))(x)
    x = layers.BatchNormalization(axis=3,name="norm"+str(i*2+18))(x)
    x=layers.LeakyReLU(alpha=0.1)(x)


x=layers.AvgPool2D(pool_size=(2,2),strides=1,padding="same")(x)
x=layers.Flatten()(x)
x=layers.Dense(1000,use_bias=True,activation=None,name="other")(x)
x=layers.Softmax()(x)

model=models.Model(input=xInput,output=x)
model.summary()



#Adam,Adadelta,Adagrad,Adanax,Nadam,


testdata=np.load("Imagenet_train.npy")
testlabel=np.load("label_train.npy")

validdata=np.load("Imagenet_valid.npy")
validlabel=np.load("label_valid.npy")

print(testdata.shape)
print(testlabel.shape)


model.compile(loss=losses.categorical_crossentropy,optimizer="Adam",metrics=["categorical_accuracy"])
callback=callbacks.EarlyStopping(min_delta=0.00001,patience=20,restore_best_weights=True)
history=model.fit(testdata,testlabel,epochs=150,batch_size=64,validation_data=(validdata,validlabel),callbacks=[callback])
model.save_weights("pre_train.h5")

from matplotlib import pyplot as plt




plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model categorical_accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
