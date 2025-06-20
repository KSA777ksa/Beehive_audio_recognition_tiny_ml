import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


def compare_audio_files(file1, file2, model_path):
    # Загрузка и обработка обоих файлов
    def process_file(file):
        y, sr = librosa.load(file, sr=16000, duration=3.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=160, center=False)

        # Фиксация размера и нормализация
        if mfcc.shape[1] < 300:
            mfcc = np.pad(mfcc, ((0, 0), (0, 300 - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :300]

            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        return mfcc

    # Получение MFCC для обоих файлов
    mfcc1 = process_file(file1)
    mfcc2 = process_file(file2)

    # Загрузка модели
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Функция для получения предсказаний и активаций
    def get_model_output(mfcc):
        input_data = np.expand_dims(mfcc, axis=(0, -1)).astype(np.float32)

        # Квантование входных данных
        if input_details['dtype'] == np.int8:
            scale, zero_point = input_details['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])

        # Деквантование выхода
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        return output[0], input_data

    # Получение результатов для обоих файлов
    output1, input1 = get_model_output(mfcc1)
    output2, input2 = get_model_output(mfcc2)

    # Анализ различий
    def print_comparison():
        # 1. Сравнение MFCC
        mfcc_diff = np.mean(np.abs(mfcc1 - mfcc2))
        print(f"\nMFCC Mean Absolute Difference: {mfcc_diff:.4f}")
        print(f"Cosine Similarity between MFCCs: {1 - cosine(mfcc1.flatten(), mfcc2.flatten()):.4f}")

        # 2. Сравнение выходов модели
        prob_diff = np.abs(output1 - output2)
        print(f"\nModel Output Differences:")
        print(f"Bee probability: {output1[0]:.4f} vs {output2[0]:.4f} (Δ={prob_diff[0]:.4f})")
        print(f"Non-bee probability: {output1[1]:.4f} vs {output2[1]:.4f} (Δ={prob_diff[1]:.4f})")

        # 3. Визуализация MFCC
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        librosa.display.specshow(mfcc1, x_axis='time')
        plt.colorbar()
        plt.title('MFCC File 1')

        plt.subplot(1, 3, 2)
        librosa.display.specshow(mfcc2, x_axis='time')
        plt.colorbar()
        plt.title('MFCC File 2')

        plt.subplot(1, 3, 3)
        librosa.display.specshow(mfcc1 - mfcc2, x_axis='time')
        plt.colorbar()
        plt.title('MFCC Difference')

        plt.tight_layout()
        plt.savefig('mfcc_comparison.png')
        print("\nSaved MFCC comparison plot to 'mfcc_comparison.png'")

        # 4. Анализ распределения значений
        print("\nInput Data Statistics:")
        print(f"File 1 - Mean: {input1.mean():.4f}, Std: {input1.std():.4f}")
        print(f"File 2 - Mean: {input2.mean():.4f}, Std: {input2.std():.4f}")

        # 5. Сравнение спектральных характеристик
        spectral_centroid1 = librosa.feature.spectral_centroid(y=librosa.load(file1, sr=16000)[0])
        spectral_centroid2 = librosa.feature.spectral_centroid(y=librosa.load(file2, sr=16000)[0])
        print(f"\nSpectral Centroid Difference: {np.mean(np.abs(spectral_centroid1 - spectral_centroid2)):.4f}")

    return print_comparison


# Пример использования
model_path = r"/main_train/models/bee_detector.tflite"
file1 = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\test_dataset\good_dataset_bee\zvuk-pchel-na-paseke.wav"
file2 = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\test_dataset\good_dataset_bee\zhuzhzhanie-pchel.wav"

comparison_func = compare_audio_files(file1, file2, model_path)
comparison_func()