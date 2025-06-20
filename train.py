import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Конфигурация
DATASET_PATH = r"D:\PyCharmProjects\TinyML_bee_recognition\final_dataset_optimal"
MODEL_SAVE_PATH = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\models"
INPUT_SHAPE = (13, 300, 1)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_dataset(dataset_path):
    classes = ['bee', 'non_bee']
    X = []
    y = []

    # Параметры MFCC
    TARGET_SHAPE = (13, 300)  # Используем константу для всех операций

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

        print(f"Loading {class_name} samples...")
        for f in tqdm(files):
            file_path = os.path.join(class_dir, f)
            try:
                y_audio, sr = librosa.load(file_path, sr=16000, duration=3.0)

                # Генерация MFCC с использованием TARGET_SHAPE
                mfcc = librosa.feature.mfcc(
                    y=y_audio,
                    sr=sr,
                    n_mfcc=TARGET_SHAPE[0],
                    n_fft=2048,
                    hop_length=160,
                    center=False
                )

                # Фиксация размера через TARGET_SHAPE
                if mfcc.shape[1] < TARGET_SHAPE[1]:
                    mfcc = np.pad(mfcc, ((0, 0), (0, TARGET_SHAPE[1] - mfcc.shape[1])))
                else:
                    mfcc = mfcc[:, :TARGET_SHAPE[1]]

                # Нормализация и добавление размерностей
                # mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
                X.append(mfcc.reshape(*TARGET_SHAPE, 1))  # Используем распаковку
                y.append(class_idx)

            except Exception as e:
                print(f"Error in {file_path}: {str(e)}")
                continue

    # Преобразование в numpy arrays
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, 2)

    # Рассчитываем глобальные параметры нормализации по всему датасету
    global_mean = np.mean(X)
    global_std = np.std(X)
    print(f"[DEBUG] Global mean: {global_mean:.6f}, Global std: {global_std:.6f}")

    # Применяем глобальную нормализацию ко всем данным
    X = (X - global_mean) / global_std

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_model():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE, name='input_layer'),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu', kernel_regularizer='l2'),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, 'best_model.keras'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            checkpoint,
            callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
        ],
        verbose=1
    )
    return history


def quantize_model():
    model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'best_model.keras'))

    def representative_dataset():
        for i in range(100):
            sample = X_train[i].reshape(1, 13, 300, 1).astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(os.path.join(MODEL_SAVE_PATH, 'bee_detector.tflite'), 'wb') as f:
        f.write(tflite_model)


def evaluate_model():
    model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'best_model.keras'))
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png'))

    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Загрузка данных
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(DATASET_PATH)

    # Обучение
    model = create_model()
    train_model(model, X_train, y_train, X_val, y_val)

    # Квантование
    quantize_model()

    # Оценка
    evaluate_model()

    print("Pipeline completed! Models saved in:", MODEL_SAVE_PATH)