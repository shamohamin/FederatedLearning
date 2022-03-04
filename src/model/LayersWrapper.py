from tensorflow import keras
from functools import partial


ReluDenseLayer = partial(
    keras.layers.Dense, kernel_initializer="he_normal",
    activation="relu", kernel_regularizer=keras.regularizers.l2())

ReluConvLayer = partial(
    keras.layers.Conv2D, activation="relu", kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2())


def load_default_model(input_shape: list, output_units: int):
    inputs = keras.layers.Input(shape=(*input_shape,))

    layers = [
        ReluConvLayer(filters=64, strides=4, kernel_size=8),
        ReluConvLayer(filters=32, strides=2, kernel_size=4),
        keras.layers.BatchNormalization(),
        ReluConvLayer(filters=16, strides=1, kernel_size=2),
        keras.layers.Flatten(),
        ReluDenseLayer(units=512),
        ReluDenseLayer(units=256),
        ReluDenseLayer(units=output_units, activation="softmax")
    ]
    
    inp = layers[0](inputs)
    
    for layer in layers[1:]:
        inp = layer(inp)
    
    model = keras.models.Model(inputs=inputs, outputs=inp)
    model.summary()
    
    return model
