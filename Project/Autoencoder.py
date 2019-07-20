from keras.layers import Input, Dense
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import Conv2D, Flatten
from keras.models import Model
from keras import backend as K



def create_encoder_model(input_shape,kernel_size,latent_dim,layer_filters):
    input_images = Input(shape=input_shape,name='encoder_input')

    x = input_images

    for filters in layer_filters:
        x = Conv2D(filters = filters,kernel_size=kernel_size,strides =2,activation = 'relu',padding ='same') (x)

    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim,name='latent_vector')(x)
    instantiate_encoder_model(input_images,latent)

def instantiate_encoder_model(input_images,latent):
    encoder = Model(input_images,latent,name ='encoder')
    encoder.summary()

def create_decoder_model(kernel_size,latent_dim,shape,layer_filters,x):
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1]*shape[2], shape[3])(latent_inputs)
    x = Reshape((shape[1],shape[2],shape[3]))(x)

    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters = filters, kernel_size = kernel_size,strides=2, activtion = 'relu', padding ='same')(x)
    
    output_images = Conv2DTranspose(filters = 1, kernel_size = kernel_size, padding='same',activation = 'sigmpod',name = 'decoder_output')(x)
    instantiate_decoder_model(latent_inputs,output_images)

def instantiate_decoder_model(latent_inputs,output_images):
    decoder = Model(latent_inputs,output_images,name = 'decoder')
    decoder.summary()

def instantiate_autoencoder_model(input_images,decoder='decoder',encoder='encoder'):
    autoencoder = Model(input_images,decoder(encoder(input_images)),name = 'autoencoder')
    autoencoder.summary()
    train_autoencoder(autoencoder)

def train_autoencoder(autoencoder,batch_size,x_train_gray,x_train,x_test_gray,x_test):
    autoencoder.compile(loss = 'mse',optimizer ='adam')
    autoencoder.fit(x_train_gray,x_train,validation_data = (x_test_gray,x_test), epochs = 30, batch_size = batch_size)
    x_decoded = autoencoder.predict(x_test_gray)



