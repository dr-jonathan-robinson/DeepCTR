import tensorflow as tf

"""
Install Tensorflow on Apple Silicon:
python -m pip install tensorflow
python -m pip install tensorflow-metal
"""

if len(tf.config.list_physical_devices('GPU')) == 0:
    print('No GPU detected')
else:
    print(tf.config.list_physical_devices('GPU'))

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)
