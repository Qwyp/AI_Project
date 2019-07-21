from keras.layers import Input, Dense
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import Conv2D, Flatten
from keras.models import Model
from keras import backend as K
import Plot as plot

def build_autoencoder_model(batch_size,input_shape,kernel_size,latent_dim,layer_filters,x_train,x_test,img_rows,img_cols,channels):
    # 1. Build encoder model
    input_images = Input(shape=input_shape,name='encoder_input')
    x = input_images

    for filters in layer_filters:
        x = Conv2D(filters = filters,kernel_size=kernel_size,strides =2,activation = 'relu',padding ='same') (x)

    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim,name='latent_vector')(x)
    # 2. Instantiate Encoder Model
    encoder = Model(input_images,latent,name ='encoder')
    encoder.summary()
    
    # 3. Build decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1]*shape[2], shape[3])(latent_inputs)
    x = Reshape((shape[1],shape[2],shape[3]))(x)

    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters = filters, kernel_size = kernel_size,strides=2, activtion = 'relu', padding ='same')(x)
    
    output_images = Conv2DTranspose(filters = 1, kernel_size = kernel_size, padding='same',activation = 'sigmpod',name = 'decoder_output')(x)
    
    # 4. Instantiate decoder Model
    decoder = Model(latent_inputs,output_images,name = 'decoder')
    decoder.summary()
    # 5. Instantiate Autoencoder Model
    # Reshape Images
    # normalize output train and test color images
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # normalize input train and test grayscale images
    x_train_gray = x_train_gray.astype('float32') / 255
    x_test_gray = x_test_gray.astype('float32') / 255
    # reshape images to row x col x channel for CNN output/validation
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    # reshape images to row x col x channel for CNN input
    x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows,img_cols, 1)
    x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

    autoencoder = Model(input_images,decoder(encoder(input_images)),name = 'autoencoder')
    autoencoder.summary()
    # 6. Train Autoencoder Model
    autoencoder.compile(loss = 'mse',optimizer ='adam')
    autoencoder.fit(x_train_gray,x_train,validation_data = (x_test_gray,x_test), epochs = 30, batch_size = batch_size)
    x_decoded = autoencoder.predict(x_test_gray)

    plot.display_colorized_predicted_images(x_decoded,x_train.shape[1],x_train.shape[2],x_train.shape[3],'saved_images')

"""
def create_encoder_model(input_shape,kernel_size,latent_dim,layer_filters):
    input_images = Input(shape=input_shape,name='encoder_input')

    x = input_images

    for filters in layer_filters:
        x = Conv2D(filters = filters,kernel_size=kernel_size,strides =2,activation = 'relu',padding ='same') (x)

    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim,name='latent_vector')(x)
    instantiate_encoder_model(input_images,latent)
    create_decoder_model(kernel_size,latent_dim,shape,layer_filters,x)
"""
def instantiate_encoder_model(input_images,latent):
    encoder = Model(input_images,latent,name ='encoder')
    encoder.summary()

"""
def create_decoder_model(kernel_size,latent_dim,shape,layer_filters,x):
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1]*shape[2], shape[3])(latent_inputs)
    x = Reshape((shape[1],shape[2],shape[3]))(x)

    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters = filters, kernel_size = kernel_size,strides=2, activtion = 'relu', padding ='same')(x)
    
    output_images = Conv2DTranspose(filters = 1, kernel_size = kernel_size, padding='same',activation = 'sigmpod',name = 'decoder_output')(x)
    instantiate_decoder_model(latent_inputs,output_images)
"""
def instantiate_decoder_model(latent_inputs,output_images):
    decoder = Model(latent_inputs,output_images,name = 'decoder')
    decoder.summary()

def instantiate_autoencoder_model(input_images,decoder='decoder',encoder='encoder'):
    autoencoder = Model(input_images,decoder(encoder(input_images)),name = 'autoencoder')
    autoencoder.summary()

def train_autoencoder(autoencoder,batch_size,x_train_gray,x_train,x_test_gray,x_test):
    autoencoder.compile(loss = 'mse',optimizer ='adam')
    autoencoder.fit(x_train_gray,x_train,validation_data = (x_test_gray,x_test), epochs = 30, batch_size = batch_size)
    x_decoded = autoencoder.predict(x_test_gray)


############################
def reshape_images_output_train(x_train,img_rows,img_cols):
    x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    return x_train

def reshape_images_output_test(x_test,img_rows,img_cols):
    x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    return x_test

def reshape_images_input_train(x_train_gray,img_rows,img_cols):
    x_train_gray = x_train_gray.reshape(x_train_gray.shape[0],img_rows,img_cols,1)
    return x_train_gray

def reshape_images_input_test(x_test_gray,img_rows,img_cols):
    x_test_gray = x_test_gray.reshape(x_test_gray.shape[0],img_rows,img_cols,1)
    return x_test_gray



