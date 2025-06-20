import os
import librosa
import soundfile as sf
from tqdm import tqdm


def process_audio_file(input_path, output_path, target_duration=3, target_sr=16000):
    """
    Обрабатывает аудиофайл:
    1. Конвертирует в моно
    2. Ресемплинг до target_sr
    3. Обрезает до target_duration секунд
    4. Сохраняет в формате WAV с битрейтом 256 kbps
    """
    try:
        # Загрузка аудио с автоматическим определением частоты дискретизации
        audio, orig_sr = librosa.load(input_path, sr=None, mono=True)

        # Пропускаем файлы короче целевой длительности
        if len(audio) < orig_sr * target_duration:
            print(f"Файл слишком короткий: {input_path}")
            return False

        # Ресемплинг до целевой частоты
        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

        # Рассчитываем количество сэмплов для целевой длительности
        target_samples = target_sr * target_duration

        # Обрезаем до нужной длительности (первые N секунд)
        if len(audio) > target_samples:
            audio = audio[:target_samples]

        # Проверка окончательной длительности
        if len(audio) != target_samples:
            print(f"Некорректная длительность после обработки: {input_path}")
            return False

        # Сохраняем с нужными параметрами
        sf.write(output_path, audio, target_sr, subtype='PCM_16')
        return True

    except Exception as e:
        print(f"Ошибка обработки {input_path}: {str(e)}")
        return False


def process_folder(input_dir, output_dir):
    """Обрабатывает все файлы в папке"""
    processed_count = 0
    skipped_count = 0

    # Создаем список всех аудиофайлов
    file_list = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_list.append(os.path.join(root, f))

    # Обработка с прогресс-баром
    for input_path in tqdm(file_list, desc="Processing files"):
        try:
            # Создаем структуру папок
            rel_path = os.path.relpath(os.path.dirname(input_path), input_dir)
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            # Формируем выходной путь
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_subdir, os.path.splitext(filename)[0] + ".wav")

            # Обработка файла
            if process_audio_file(input_path, output_path):
                processed_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            print(f"Ошибка обработки файла {input_path}: {str(e)}")
            skipped_count += 1

    print(f"\nОбработка завершена! Успешно: {processed_count}, Пропущено: {skipped_count}")


if __name__ == "__main__":
    input_folder = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\test_dataset\raw_dataset_bee"  # Ваш исходный путь
    output_folder = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\test_dataset\good_dataset_bee"  # Ваш целевой путь

    process_folder(input_folder, output_folder)