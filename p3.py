

import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



x_train = x_train.reshape(x_train.shape[0],784)
x_test = x_test.reshape(x_test.shape[0],784)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


model = tf.keras.models.Sequential([tf.keras.layers.Dense(128 , activation='relu' , input_shape = (784,)),
                                    tf.keras.layers.Dense(10, activation = 'softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)


loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)


