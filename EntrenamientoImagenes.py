import sys  #libreria para moverse en carpetas
import os   #libreria para moverse en carpetas
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #pre prosesar imagenes de entrenamiento
from tensorflow.python.keras import optimizers # Entrenador de algoritmo
from tensorflow.python.keras.models import Sequential # Libreria para redes neurales secuenciales 
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation # 
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D # capas de convulciones
from tensorflow.python.keras import backend as K # controlar sesion de keras

K.clear_session()


data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

"""
Parameters
"""
epocas = 20 #numero de veces que iteran los datos 
altura, longitud = 100, 100 #tamaño
batch_size = 32 #numero de imagenes para enviar a procesar
pasos = 1000 # numero de veces para procesar en epocas
pasos_validacion = 200 #correr 200 pasos para saber que tan bien esta aprendiendo el algoritmo
filtrosConv1 = 32 #profundidad
filtrosConv2 = 64
tamano_filtro1 = (3,3) 
tamano_filtro2 = (2,2)
tamano_pool = (2,2)
clases = 1
lr = 0.0005


##Pre proceso de nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255, #valores de pixeles esten de 0 a 1 mas eficiente el entrenamiento 
    shear_range=0.3, #inclinador de imagenes
    zoom_range=0.3, #zoom de imagenes para el entrenamiento
    horizontal_flip=True #invertir imagenes
) 

validacion_datagen = ImageDataGenerator(
    rescale=1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(

    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

imagen_validacion = validacion_datagen.flow_from_directory(

    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

#Red convolucional

cnn = Sequential() #varias capas apiladas

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten()) #imagen profunda y pequeña queda plana
cnn.add(Dense(256, activation='relu')) #despues de aplanar la imformacion es enviada a una capa nueva
cnn.add(Dropout(0.5)) #a esta capa densa se le apaga el 50% de 256 neuronas a cada paso.
cnn.add(Dense(clases, activation='softmax')) #esta activacion es la que ayuda a predecir

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])


cnn.fit_generator(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)

dir = './modelo/'

if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5') #se guarda estructura del modelo
cnn.save_weights('./modelo/pesos.h5') #los pesos de cada capa que ya se entreno