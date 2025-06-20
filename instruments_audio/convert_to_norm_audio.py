import os
import librosa
import soundfile as sf


def process_audio_file(input_path, output_path, target_sr=16000, target_duration=10):
    """
    Обрабатывает аудиофайл:
    1. Конвертирует в моно
    2. Изменяет частоту дискретизации
    3. Обрезает до target_duration секунд
    """
    # Загружаем аудио с преобразованием в моно и целевой частотой дискретизации
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # Рассчитываем максимальное количество семплов для целевой длительности
    max_samples = target_duration * target_sr

    # Обрезаем до первых 10 секунд если нужно
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # Сохраняем обработанный файл
    sf.write(output_path, audio, target_sr, 'PCM_16')


def process_directory(input_dir, output_dir):
    """
    Обрабатывает все аудиофайлы в директории
    """
    # Создаем выходную директорию если не существует
    os.makedirs(output_dir, exist_ok=True)

    # Обрабатываем все файлы во входной директории
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing: {filename}")
            try:
                process_audio_file(input_path, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_folder = "Kaggle_Audio_Bee_NoBee"  # Путь к исходной папке с аудио
    output_folder = "Kaggle_Audio_Bee_NoBee_norm"  # Путь к папке для обработанных файлов

    process_directory(input_folder, output_folder)
    print("Processing completed!")