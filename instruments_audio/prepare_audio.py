import os
from pydub import AudioSegment


def convert_mp3_to_wav(mp3_file, output_dir):
    """Конвертирует mp3 файл в wav формат."""
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        wav_file = os.path.join(output_dir, os.path.basename(mp3_file).replace(".mp3", ".wav"))
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        print(f"Ошибка при конвертации {mp3_file}: {e}")
        return None


def split_audio(audio_file, segment_duration=10, output_dir=None):
    """Делит аудиофайл на отрезки по заданной длительности (в секундах)."""
    try:
        audio = AudioSegment.from_wav(audio_file)
        duration_ms = len(audio)
        segment_duration_ms = segment_duration * 1000  # длительность отрезка в миллисекундах

        os.makedirs(output_dir, exist_ok=True)

        # Разделение на отрезки
        for i in range(0, duration_ms, segment_duration_ms):
            segment = audio[i:i + segment_duration_ms]
            segment_filename = os.path.join(output_dir,
                                            f"{os.path.basename(audio_file).replace('.wav', '')}_part_{i // segment_duration_ms + 1}.wav")
            segment.export(segment_filename, format="wav")
            print(f"Отрезок сохранен как: {segment_filename}")
    except Exception as e:
        print(f"Ошибка при делении файла {audio_file}: {e}")


def process_directory(input_dir, output_dir):
    """Обрабатывает все mp3 файлы в указанной директории: конвертирует в wav и делит на отрезки."""
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".mp3"):
            mp3_file_path = os.path.join(input_dir, file_name)

            # Преобразуем mp3 в wav
            wav_file = convert_mp3_to_wav(mp3_file_path, output_dir)

            if wav_file:
                # Делим wav файл на отрезки по 10 секунд
                split_audio(wav_file, segment_duration=10, output_dir=output_dir)


if __name__ == "__main__":
    input_directory = "Dataset_mp3/No_Bee2"  # Путь к директории с mp3 файлами
    output_directory = "Dataset_for_Test/No_Bee"  # Путь к директории, куда будут сохраняться файлы wav

    process_directory(input_directory, output_directory)
