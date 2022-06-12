# Import Modules
import numpy as np
import tensorflow as tf
import datetime
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
from numba import cuda
import os





# Heteroscedasktic loss function

def heteroscedastic_loss(true, pred):
    """ Heteroskedastic loss function."""
    params = pred.shape[1] // 2
    point = pred[:, :params]
    var = pred[:, params:]
    precision = 1 / var
    tf.autograph.experimental.do_not_convert
    return keras.backend.sum(precision * (true - point) ** 2. + keras.backend.log(var), -1)

# Create Model ( will be changed for taking on the fly generated tensors from M3_generator function
def create_model(n_params, channels, filters, batchnorm=False, training=True):
    """Creates 1DConv model."""

    inp = keras.Input(shape=(None, channels))
    x = inp
    x = keras.layers.Conv1D(filters[0], kernel_size=1, strides=1, activation='relu')(x)

    for f in filters[1:]:
        x = keras.layers.Conv1D(f, kernel_size=3, strides=2, activation='relu')(x)
        # x = keras.layers.Dropout(rate=0.2)(x,training=True)
        if batchnorm:
            x = keras.layers.BatchNormalization()(x, training=training)

    x = keras.layers.GlobalAveragePooling1D()(x)
    mean = keras.layers.Dense(n_params)(x)
    var = keras.layers.Dense(n_params, activation='softplus')(x)
    out = keras.layers.Concatenate()([mean, var])
    model = keras.Model(inp, out)
    return model

# Create Logs for Tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# generate adhoc tensors
dataset = tf.data.Dataset.from_generator(
    generator=generate_M3,
    args=[batchsize],  # batch_size
    output_shapes=(
        tf.TensorShape([batchsize, 1000, 5]),
        tf.TensorShape([1, ])
    ),
    output_types=(tf.float32, tf.int32))



# Training
keras.backend.clear_session()
deepInference = create_model(n_params=len(NAMES), channels=nchannels, filters=filters)
deepInference.summary()

deepInference.compile(optimizer=optimizer, loss=heteroscedastic_loss, metrics=['accuracy'])
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=2)
checkpointer = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=False, verbose=2)

# Fit Model

 history = deepInference.fit(M3_data, epochs=10
                                epochs=10,
                                validation_split=0.01,
                                callbacks=[earlystopper, checkpointer,tensorboard_callback])


# Make Predictions from Test Data
preds = deepInference.predict(####)








# Scratch Code
#
# dataset = tf.data.Dataset.from_generator(
#     generator=generate_tmvn,
#     args=[128],  # batch_size
#     output_shapes=(
#         tf.TensorShape([128, 1000, 3]),
#         tf.TensorShape([1, ])
#     ),
#     output_types=(tf.float32, tf.int32))
#
# model = models.Sequential([
#     layers.Dense(4),
#     layers.Conv1D(filters=8, kernel_size=3, padding="same"),
#     layers.Flatten(),
#     layers.Dense(1),
# ])
# model.compile(loss="mean_squared_error")
# model.build(input_shape=(1, 1000, 3))
# model.summary()
#
# model.fit(dataset, epochs=10)
