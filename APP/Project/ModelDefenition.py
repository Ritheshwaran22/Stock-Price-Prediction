"""ModelDefenition.py: Defines Bidirectional LSTM model with correct layer sequence handling"""

try:
    import tensorflow as tf
except ImportError:
    tf = None
from . import constants

def defineModel(XTrain, noOfLSTMUnits=128, noOfDenseUnits=64, noOfBiDirectionalLayers=1,
                noOfDenseLayers=1, dropoutRate=0.01, returnSequences=False):
    """
    Defines a Bidirectional LSTM Sequential model.
    Properly handles return_sequences so intermediate recurrent layers return sequences,
    while the final recurrent layer outputs a 2D tensor fed directly into Dense layers.
    """
    if tf is None:
        raise ImportError("TensorFlow is not installed in the current environment. Please install tensorflow to define models.")
    model = tf.keras.Sequential()

    # Determine sequence return behavior:
    # If there are multiple LSTM layers, intermediate layers must return sequences.
    # The last LSTM layer must have return_sequences=False.
    first_layer_return_seq = True if noOfBiDirectionalLayers > 1 else False

    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=noOfLSTMUnits,
                return_sequences=first_layer_return_seq
            ),
            input_shape=(XTrain.shape[1], XTrain.shape[2])
        )
    )

    # Intermediate and final LSTM layers
    for i in range(1, noOfBiDirectionalLayers):
        is_last_lstm = (i == noOfBiDirectionalLayers - 1)
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=noOfLSTMUnits,
                    return_sequences=not is_last_lstm
                )
            )
        )

    # Dense layers
    for i in range(noOfDenseLayers):
        model.add(tf.keras.layers.Dense(units=noOfDenseUnits, activation='relu'))
        if dropoutRate > 0:
            model.add(tf.keras.layers.Dropout(rate=dropoutRate))

    # Single continuous price regression output
    model.add(tf.keras.layers.Dense(units=1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=constants.LEARNING_RATE)
    model.compile(
        loss=constants.LOSS_FUNCTION,
        optimizer=optimizer,
        metrics=constants.METRICS
    )

    return model
