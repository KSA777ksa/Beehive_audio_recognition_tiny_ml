import os
import librosa
import soundfile as sf

def split_audio(input_folder, output_folder, segment_duration=10):
    """
    Делит все аудиофайлы в папке на сегменты заданной длительности (в секундах).

    :param input_folder: Папка с исходными аудиофайлами.
    :param output_folder: Папка для сохранения сегментированных файлов.
    :param segment_duration: Длительность каждого сегмента (в секундах).
    """
    os.makedirs(output_folder, exist_ok=True)  # Создаём выходную папку, если её нет

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            input_path = os.path.join(input_folder, file_name)

            try:
                # Загружаем аудио
                audio, sr = librosa.load(input_path, sr=None, mono=True)

                # Рассчитываем количество сегментов
                segment_samples = int(sr * segment_duration)  # Количество сэмплов в одном сегменте
                total_segments = len(audio) // segment_samples

                for i in range(total_segments):
                    start_sample = i * segment_samples
                    end_sample = start_sample + segment_samples
                    segment_audio = audio[start_sample:end_sample]

                    # Создаём имя для сегмента
                    segment_file_name = f"{os.path.splitext(file_name)[0]}_segment{i + 1}.wav"
                    output_path = os.path.join(output_folder, segment_file_name)

                    # Сохраняем сегментированный файл
                    sf.write(output_path, segment_audio, sr)
                    print(f"Сохранён сегмент: {output_path}")
            except Exception as e:
                print(f"Ошибка обработки файла {file_name}: {e}")

# Пример использования
input_folder = "Dataset_bee/sound_files/sound_files"  # Укажите путь к папке с аудиофайлами
output_folder = "Dataset_bee_10_sec"  # Укажите путь к папке для сохранения сегментов
split_audio(input_folder, output_folder)
