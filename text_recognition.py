import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training the model...")
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


print("Evaluating the model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


predictions = model.predict(x_test[:5])
print("Sample predictions:")
for i in range(5):
    print("Predicted value:", tf.argmax(predictions[i]).numpy())
    print("True value:", y_test[i])
    plt.imshow(x_test[i], cmap='gray')
    plt.show()
