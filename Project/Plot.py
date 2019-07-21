import matplotlib.pyplot as plt
import numpy as np

# Used source: https://blog.keras.io/building-autoencoders-in-keras.html

""" This file is plotting the different outcomes
- Original Color Images
- Grayscaled Images
- Predicted Color from Images """

def display_colorized_predicted_images(x_decoded,img_rows,img_cols,channels,imgs_dir):
    # display first 100 images
    imgs = x_decoded[:100]
    imgs = imgs.reshape((10,10,img_rows,img_cols,channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Predicted Colors for Images')
    plt.imshow(imgs, interpolation='none')
    plt.savefig('%s/colorized_predicted.png' % imgs_dir)


def display_images_original_color(x_test,img_rows,img_cols,channels,imgs_dir):
    # display first 100 images
    imgs = x_test[:100]
    imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Cororized Test Images')
    plt.imshow(imgs, interpolation='none')
    plt.savefig('%s/test_color.png' % imgs_dir)
    plt.show()



def convert_rgb_to_gray(rgb):
    # convert from color image (RGB) to grayscale
    # source: opencv.org
    # grayscale = 0.299*red + 0.587*green + 0.114*blue
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def display_images_grayscale(x_test_gray,img_rows,img_cols,imgs_dir):
    # display first 100 images in grayscaled

    imgs = x_test_gray[:100]
    imgs = imgs.reshape((10, 10, img_rows, img_cols))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Grayscaled Input Images')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.savefig('%s/test_gray.png' % imgs_dir)
    plt.show()
