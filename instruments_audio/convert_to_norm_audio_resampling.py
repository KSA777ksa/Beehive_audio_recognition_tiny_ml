import os
import librosa
import soundfile as sf
from tqdm import tqdm  # Для прогресс-бара


def resample_audio(input_path, output_path,
                   original_sr=22050,
                   target_sr=16000,
                   target_duration=10):
    """
    Конвертирует аудиофайл:
    1. Ресемплинг 22050 → 16000 Гц
    2. Конвертация в моно
    3. Обрезка до 10 секунд
    """
    try:
        # Загружаем с исходной частотой 22050 Гц
        audio, _ = librosa.load(input_path, sr=original_sr, mono=True)

        # Ресемплинг до 16000 Гц
        audio_resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

        # Обрезка до первых 10 секунд
        max_samples = target_duration * target_sr
        if len(audio_resampled) > max_samples:
            audio_resampled = audio_resampled[:max_samples]

        # Сохраняем с новыми параметрами
        sf.write(output_path, audio_resampled, target_sr, subtype='PCM_16')

    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")


def process_folder(input_dir, output_dir):
    """Обрабатывает все файлы в папке"""
    os.makedirs(output_dir, exist_ok=True)

    # Получаем список файлов с поддержкой вложенных папок
    file_list = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                file_list.append((root, f))

    # Обработка с прогресс-баром
    for root, filename in tqdm(file_list, desc="Processing files"):
        input_path = os.path.join(root, filename)
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, filename)

        resample_audio(input_path, output_path)


if __name__ == "__main__":
    input_folder = "Dataset_bee_10_sec"  # Замените на свой путь
    output_folder = "Dataset_bee_10_sec_norm"  # Замените на свой путь

    process_folder(input_folder, output_folder)
    print("\nКонвертация завершена!")