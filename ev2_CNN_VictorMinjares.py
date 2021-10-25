#!/usr/bin/env python
# coding: utf-8

# Red neuronal convolucinal para clasificacion de imagenes.

# Guardara el modelo en un .h5 y nos dara graficas del proceso
# de entrenamiento comparando el accuracy y loss de los datos
# de validacion y entrenamiento.
# Tendra hiperparametros los cuales se pondran modificar desde
# un archivo externo .slrm.
# El dataset a utilizar sera obtenido directamente de kaggle.

 
# Paqueterias a utilizar:

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import argparse 
import matplotlib.pyplot as plt
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard

# Argumentos:

parser = argparse.ArgumentParser(description='definir hiperparametros red neuronal')

parser.add_argument('augmentation', type=str, help='Augmentation: yes/no ')

parser.add_argument('dropout', type=str, help='Dropout: yes/no ')

parser.add_argument('optimizer', type=str, help='Optimizer: adam,sgd,... ')

parser.add_argument('activation', type=str, help='Activation: relu,elu,sigmoid,... ')

parser.add_argument('batchsize', type=int, help='batch size: 20,30,46,... ')

parser.add_argument('dimconv', type=int, help='Dimension of convolutional layers: 2,3,4,... ')

parser.add_argument('dimpool', type=int, help='Dimension of pooling layers: 2,3,4,... ')

parser.add_argument('numlay', type=int, help='Choose between 3 o 5 layers in the CNN: 3/5 ')

args = parser.parse_args()

print("Augmentation : ",args.augmentation)

print("Dropout : ",args.dropout)

print("The optimizer used is : ",args.optimizer)

print("The activation used is : ",args.activation)

print("The batch size used is : ",args.batchsize)

print("The dimension of convolutional layers is : ",args.dimconv)

print("The dimension of pooling layers is : ",args.dimpool)

print("Number of layers : ",args.numlay)

# Lineas exclusivas si cuentas con una tarjeta GPU (Nvidia) configurada (CUDA+cudnn) 

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
#print("GPUs: " + gpu_devices[0])

gpus = tf.test.gpu_device_name()
print("GPUs: " + gpus)



# Descargamos las imagenes:

# Si ya tenemos corrimos el programa una vez no es necesario seguir teniendo las dos primeras lineas.

#os.system(" kaggle datasets download -d crowww/a-large-scale-fish-dataset -p $PWD ")
#os.system( " unzip a-large-scale-fish-dataset.zip ")
data_dir = pathlib.Path("Fish_Dataset/Fish_Dataset")

# Numero de clases a clasificar
num_classes = 9


# Vemos cuantas imagenes tiene el dataset


image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)



# Creamos el dataset:

# Parametros para usar en los dataset (entrenamiento, validacion):


batch_size = args.batchsize 
# Tamano de la imagen 180x180 pixeles
img_height = 180
img_width = 180


# Datos de entrenamiento 80% del total


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Datos de validacion 20% del total


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# Vemos las clases de nuestro dataset


class_names = train_ds.class_names
print(class_names)

# Verificamos si las dimensiones de nuestro dataset son las correctas

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Con esto mejoramos el rendimiento, para que no sucedan cuellos de botella en el entrenamiento
# o cuando el dataset es muy grande y no cabe en la memoria.

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Creamos el modelo:

###############################################################################################################

# If para saber si se quiere usar 3 o 5 capas en la red neuronal

if (args.numlay == 3):

# If's para saber si se usaran los metodos de dropout
# y/o augmentation

  if (args.augmentation == 'no') and (args.dropout == 'no'):

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height,img_width,3)),
        layers.Conv2D(16, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Conv2D(32, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Conv2D(64, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Flatten(),
        layers.Dense(128, activation=args.activation),
        layers.Dense(num_classes)
        ])
    
    
    
  elif (args.augmentation == 'no') and (args.dropout == 'yes'):
  
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height,img_width,3)),
        layers.Conv2D(16, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Conv2D(32, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Conv2D(64, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Dropout(0.2),
        layers.Flatten(), 
        layers.Dense(128, activation=args.activation),
        layers.Dense(num_classes)
        ])
    
    
    
  elif (args.augmentation == 'yes') and (args.dropout == 'no'):
  
    data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                     input_shape=(img_height, 
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        ]
      )
  
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Conv2D(32, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Conv2D(64, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Flatten(),
        layers.Dense(128, activation=args.activation),
        layers.Dense(num_classes)
        ])


  else:
  
    data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                     input_shape=(img_height, 
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        ]
      )
  
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Conv2D(32, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Conv2D(64, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation=args.activation),
        layers.Dense(num_classes)
        ])

elif (args.numlay == 5):

# If's para saber si se usaran los metodos de dropout
# y/o augmentation

  if (args.augmentation == 'no') and (args.dropout == 'no'):

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height,img_width,3)),
        layers.Conv2D(16, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(32, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(64, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(128, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(256, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation=args.activation),
        layers.Dense(num_classes)
        ])
    


  elif (args.augmentation == 'no') and (args.dropout == 'yes'):
  
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height,img_width,3)),
        layers.Conv2D(16, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(32, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(64, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(128, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(256, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Dropout(0.2),
        layers.Flatten(), 
        layers.Dense(128, activation=args.activation),
        layers.Dense(num_classes)
        ])



  elif (args.augmentation == 'yes') and (args.dropout == 'no'):
  
    data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                     input_shape=(img_height, 
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        ]
      )
  
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(32, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(64, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(128, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(256, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation=args.activation),
        layers.Dense(num_classes)
        ])


  else:
  
    data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                     input_shape=(img_height, 
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        ]
      )
  
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(32, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(64, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(128, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Conv2D(256, kernel_size=(args.dimconv,args.dimconv), padding='same', activation=args.activation),
        layers.MaxPooling2D(pool_size=(args.dimpool,args.dimpool), padding='same'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation=args.activation),
        layers.Dense(num_classes)
        ])

#################################################################################################################


#  Compilamos y entrenamos el modelo


model.compile(optimizer=args.optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Resumen del modelo:


model.summary()


# Callbacks:


my_callbacks = [
    TensorBoard(log_dir="tensorboard/cnn_fish",
                write_graph=True,
                update_freq="epoch")
]


# Entrenamiento:


epochs = 15 # para alcanzar un accuracy bueno
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=my_callbacks
)


# Visualizamos los resultados del entrenamiento:



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.show()
plt.savefig("graficos_"+args.augmentation+"_"+args.dropout+"_"+args.optimizer+"_"+args.activation+"_"+str(args.batchsize)+"_"+str(args.dimconv)+"_"+str(args.dimpool)+"_"+str(args.numlay)+".png")



#  Prediccion:

# Usamos una imagen independiente para ver si el modelo es bueno.


fish_path = '/LUSTRE/home/ccd/Victor_Minjares/evaluacion2/test_fish/sea_bass.png'

img = keras.preprocessing.image.load_img(
    fish_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Creamos un batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "Esta imagen lo mas probable es que pertenezca a {} con un {:.2f} porciento de exactitud."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


model.save("model_"+args.augmentation+"_"+args.dropout+"_"+args.optimizer+"_"+args.activation+"_"+str(args.batchsize)+"_"+str(args.dimconv)+"_"+str(args.dimpool)+"_"+str(args.numlay)+".h5")
