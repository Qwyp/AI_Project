import os

def save_datamodel():
    save_folder = os.path.join(os.getcwd(),'saved_datamodels')
    model_name = 'datamodel_predicted_color.{epoch:03d}.h5'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    filepath = os.paht.join(save_folder,model_name)

def save_image():
    save_folder = os.path.join(os.getcwd(),'saved_images')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    filepath = os.paht.join(save_folder)