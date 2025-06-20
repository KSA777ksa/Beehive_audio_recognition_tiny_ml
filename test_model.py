import numpy as np
import librosa
import tensorflow as tf

# Конфигурация (должна совпадать с обучением)
GLOBAL_MEAN = 6.025622  # Значения из вашего обучения
GLOBAL_STD = 47.826233  # [DEBUG] Global mean: 6.025622, Global std: 47.826233
TARGET_SAMPLES = 16000 * 3  # 3 секунды при 16 кГц


def predict_audio(audio_path, tflite_model_path):
    # 1. Загрузка и проверка длины аудио
    y, sr = librosa.load(audio_path, sr=16000, duration=3.0)

    # Фиксация длины как в обучении
    if len(y) < TARGET_SAMPLES:
        y = librosa.util.fix_length(y, size=TARGET_SAMPLES)
    else:
        y = y[:TARGET_SAMPLES]

    # 2. Извлечение MFCC с параметрами из обучения
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=160,
        center=False
    )

    # 3. Фиксация размера MFCC
    if mfcc.shape[1] < 300:
        mfcc = np.pad(mfcc, ((0, 0), (0, 300 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :300]

    # 4. Глобальная нормализация (как в обучении)
    mfcc = (mfcc - GLOBAL_MEAN) / GLOBAL_STD

    # 5. Подготовка входных данных
    input_data = np.expand_dims(mfcc, axis=(0, -1)).astype(np.float32)  # (1,13,300,1)

    # 6. Загрузка и настройка интерпретатора TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 7. Квантование входных данных
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        input_data = (input_data / scale + zero_point).astype(np.int8)

    # 8. Инференс
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    # 9. Получение и деквантование выхода
    output = interpreter.get_tensor(output_details['index'])
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    # 10. Форматирование результата
    probabilities = output[0]
    return {
        "prediction": "bee" if np.argmax(probabilities) == 0 else "non_bee",
        "probabilities": {
            "bee": float(probabilities[0]),
            "non_bee": float(probabilities[1])
        },
        "debug_info": {
            "audio_length": len(y),
            "mfcc_shape": mfcc.shape,
            "input_mean": float(np.mean(input_data)),
            "input_std": float(np.std(input_data))
        }
    }


# Пример использования
if __name__ == "__main__":
    model_path = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\models\bee_detector.tflite"
    audio_file = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\test_dataset\good_dataset_bee\zvuk-pchel-na-paseke.wav"

    result = predict_audio(audio_file, model_path)

    print(f"Prediction: {result['prediction']}")
    print(
        f"Probabilities: Bee - {result['probabilities']['bee']:.4f}, Non-bee - {result['probabilities']['non_bee']:.4f}")
    print("\nDebug Info:")
    print(f"Audio length: {result['debug_info']['audio_length']} samples")
    print(f"MFCC shape: {result['debug_info']['mfcc_shape']}")
    print(f"Input data mean: {result['debug_info']['input_mean']:.4f}")
    print(f"Input data std: {result['debug_info']['input_std']:.4f}")