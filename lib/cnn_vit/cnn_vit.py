from cnn_vit.cnn_vit_config import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from myLayers.vision_transformer import add_vit


def build_model(image_size):
    augmentation = keras.Sequential(
        [
        layers.Rescaling(scale=scale),
        layers.RandomFlip(flip),
        layers.RandomRotation(rotation_factor),
        layers.RandomZoom(height_factor=zoom_height_factor, width_factor=zoom_width_factor),
        ],
        name='augmentation'
    )

    encoder = keras.Sequential(
        [
        layers.Conv2D(128, (3, 3), activation = 'relu', padding='same', kernel_initializer='random_normal'),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Conv2D(128, (3, 3), activation = 'relu', padding='same', kernel_initializer='random_normal'),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Conv2D(64, (3, 3), activation = 'relu', padding='same', kernel_initializer='random_normal'),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Conv2D(64, (3, 3), activation = 'relu', padding='same', kernel_initializer='random_normal'),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Conv2D(64, (3, 3), activation = 'relu', padding='same', kernel_initializer='random_normal'),
        layers.MaxPooling2D(pool_size = (2, 2)),
        # layers.Flatten(),
        ],
        name='encoder'
    )

    classifier = keras.Sequential(
        [
        layers.Dense(1024, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(3, activation='softmax'),
        ],
        name='classifier'
    )

    input_shape = (image_size, image_size, 1)
    inputs = layers.Input(shape=input_shape)
    augmented_transformer = augmentation(inputs)
    shared_encoded = encoder(augmented_transformer)
    features = add_vit(shared_encoded,
                patch_size=patch_size,
                input_image_size=shared_encoded.shape[1],
                transformer_layers=transformer_layers,
                num_heads=num_heads,
                projection_dim=projection_dim,
                transformer_units_rate=transformer_units_rate,
                mlp_head_units=mlp_head_units)
    # Classify outputs.
    softmax = layers.Dense(3, activation='softmax', kernel_initializer='random_normal')(features)

    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=softmax)

    return model


def build_grad_cam(model):
    output_conv_layer = model.get_layer("encoder").get_output_at(0)

    grad_model = keras.Model(
    [model.input],
    [output_conv_layer, model.output])
    return grad_model







