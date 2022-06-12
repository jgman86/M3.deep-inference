import numpy as np
import tensorflow as tf
from keras import layers, models

def generate_tmvn(batch_size=1, steps=10, size=1000, mu=np.zeros(3), cov=np.eye(3)):
    # Truncation logic / Truncated MVN instead of MVN
    i = 0
    while i < steps:
        i += 1
        batch = np.random.multivariate_normal(mean=mu, cov=cov, size=(batch_size, size))
        batch_tf = tf.convert_to_tensor(batch)  # dims: (batch_size, size, 3)
        yield (batch_tf, tf.constant([1]))  # (data, label)


dataset = tf.data.Dataset.from_generator(
    generator=generate_tmvn,
    args=[128],  # batch_size
    output_shapes=(
        tf.TensorShape([128, 1000, 3]),
        tf.TensorShape([1, ])
    ),
    output_types=(tf.float32, tf.int32))

model = models.Sequential([
    layers.Dense(4),
    layers.Conv1D(filters=8, kernel_size=3, padding="same"),
    layers.Flatten(),
    layers.Dense(1),
])
model.compile(loss="mean_squared_error")
model.build(input_shape=(1, 1000, 3))
model.summary()

model.fit(dataset, epochs=10)
