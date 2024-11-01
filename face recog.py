import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Her sınıftan seçilecek veri sayısı
samples_per_class = 200
img_size = (48, 48)
batch_size = 16
num_classes = 7  # 7 farklı sınıf var

# Verilerin bulunduğu dizinler
train_data_dir = "C:/Users/gulec/Desktop/EmotionCNN/train"
val_data_dir = "C:/Users/gulec/Desktop/EmotionCNN/test"

# Veri artırımı
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Eğitim verilerini yükleme ve sınırlama
def load_data_limited(data_dir, target_size, batch_size, samples_per_class):
    generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True,
        batch_size=batch_size)

    class_indices = generator.class_indices
    class_counts = {class_name: 0 for class_name in class_indices}

    x_data, y_data = [], []

    while any(count < samples_per_class for count in class_counts.values()):
        x_batch, y_batch = next(generator)
        for img, label in zip(x_batch, y_batch):
            label_class = np.argmax(label)
            class_name = list(class_indices.keys())[label_class]
            if class_counts[class_name] < samples_per_class:
                x_data.append(img)
                y_data.append(label)
                class_counts[class_name] += 1

            if all(count >= samples_per_class for count in class_counts.values()):
                break

    return np.array(x_data), np.array(y_data)


# Eğitim ve doğrulama verilerini sınırlı olarak yükle
x_train, y_train = load_data_limited(train_data_dir, img_size, batch_size, samples_per_class)
x_val, y_val = load_data_limited(val_data_dir, img_size, batch_size, samples_per_class)

# Modeli tanımla ve eğit
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
history = model.fit(x_train, y_train, epochs=50, batch_size=batch_size, validation_data=(x_val, y_val))

# Eğitim doğruluğu ve kaybı grafiğini çiz
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.show()

plot_training_history(history)

# Karmaşık matris (confusion matrix) oluşturma
def plot_confusion_matrix(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.title('Karmaşık Matris')
    plt.show()

plot_confusion_matrix(model, x_val, y_val)