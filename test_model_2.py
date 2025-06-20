import numpy as np
import librosa
import tensorflow as tf


def predict_audio(audio_path, tflite_model_path):
    # Загрузка и предобработка аудио
    y, sr = librosa.load(audio_path, sr=16000, duration=3.0)

    # Извлечение MFCC с теми же параметрами, что при обучении
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=160,
        center=False
    )

    # Фиксация размера (13, 300)
    if mfcc.shape[1] < 300:
        mfcc = np.pad(mfcc, ((0, 0), (0, 300 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :300]

    # Нормализация (как при обучении)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Добавление размерностей (1, 13, 300, 1)
    input_data = np.expand_dims(mfcc, axis=(0, -1)).astype(np.float32)

    # Загрузка TFLite модели
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Получение информации о входе/выходе
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Квантование входных данных (если требуется)
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        input_data = (input_data / scale + zero_point).astype(np.int8)

    # Подача данных в модель
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    # Получение и декодирование выходных данных
    output = interpreter.get_tensor(output_details['index'])

    # Деквантование выхода (если требуется)
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    probabilities = output[0]

    # Получение предсказания
    return {
        "prediction": "bee" if np.argmax(probabilities) == 0 else "non_bee",
        "probabilities": {
            "bee": float(probabilities[0]),
            "non_bee": float(probabilities[1])
        }
    }


# Пример использования
model_path = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\second_models\bee_detector.tflite"
audio_file = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\test_dataset\good_dataset_bee\zvuk-pchel-na-paseke.wav"

# Пример использования
result = predict_audio(audio_file, model_path)
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: Bee - {result['probabilities']['bee']:.4f}, Non-bee - {result['probabilities']['non_bee']:.4f}")