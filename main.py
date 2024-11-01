import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter

# Focal Loss fonksiyonu
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Küçük bir epsilon ekleyerek log hatalarını önleyelim
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.convert_to_tensor(y_true, tf.float32)

        # Kayıp hesaplama
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

# Resim boyutu, batch size ve sınıf sayısı
img_size = (48, 48)
batch_size = 32 # Batch size artırıldı
num_classes = 7  # 7 farklı sınıf var

# Verilerin bulunduğu dizinler
train_data_dir = "C:/Users/gulec/Desktop/EmotionCNN/train"
val_data_dir = "C:/Users/gulec/Desktop/EmotionCNN/test"

# Veri artırımı (Data Augmentation) - eğitim verisi için çeşitli dönüşümler
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Doğrulama verisi sadece yeniden ölçeklendirilir
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Eğitim ve doğrulama verilerini yükle (tüm verileri kullanacağız)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    batch_size=batch_size
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False,
    batch_size=batch_size
)

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

# Modeli Focal Loss ile derle
model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])

# Modeli eğit
history = model.fit(
    train_generator,
    epochs=75,
    batch_size=batch_size,
    validation_data=val_generator
)

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
def plot_confusion_matrix(model, val_generator):
    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred_classes)
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.title('Karmaşık Matris')
    plt.show()

    return y_true, y_pred_classes

y_true, y_pred_classes = plot_confusion_matrix(model, val_generator)

# Sınıflara göre doğru tahmin sayısını hesapla
def calculate_class_accuracy(y_true, y_pred_classes, num_classes):
    class_counts = Counter(y_true)
    class_correct = Counter()

    for true, pred in zip(y_true, y_pred_classes):
        if true == pred:
            class_correct[true] += 1

    class_accuracy = {i: class_correct[i] / class_counts[i] if class_counts[i] > 0 else 0 for i in range(num_classes)}
    return class_accuracy

# Sınıf doğruluklarını hesapla ve sırala
class_accuracy = calculate_class_accuracy(y_true, y_pred_classes, num_classes)
sorted_class_accuracy = sorted(class_accuracy.items(), key=lambda x: x[1])

# Sonuçları yazdır
class_names = ['Disgust', 'Fear', 'Angry', 'Surprise', 'Neutral', 'Sad', 'Happy']
print("Sınıf Doğrulukları (Küçükten Büyüğe):")
for i, accuracy in sorted_class_accuracy:
    print(f"{class_names[i]}: {accuracy:.4f}")
