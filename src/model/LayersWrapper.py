from tensorflow import keras

def load_model():
    inputs = keras.layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = keras.layers.Flatten()(layer3)

    layer5 = keras.layers.Dense(512, activation="relu")(layer4)
    action = keras.layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)
