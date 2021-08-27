import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_classes = 20
img_rows, img_cols = 32, 32
batch_size = 32

my_path = os.path.abspath(os.path.dirname(__file__))
train_data_dir = os.path.join(my_path, './simpsons/train')
validation_data_dir = os.path.join(my_path, './simpsons/validation')

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
    
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
