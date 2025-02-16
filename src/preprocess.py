from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, batch_size=64, target_size=(48, 48)):
    train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        rescale=1./255,
        validation_split=0.2
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset="training"
    )
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation"
    )
    return train_generator, validation_generator
