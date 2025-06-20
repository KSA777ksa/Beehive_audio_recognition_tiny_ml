import os
import librosa
import soundfile as sf


def split_audio(input_folder, output_folder, segment_duration=3):
    """
    Делит аудиофайлы на сегменты заданной длительности, отбрасывая остаток.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if not file_name.lower().endswith('.wav'):
            continue

        input_path = os.path.join(input_folder, file_name)

        try:
            audio, sr = librosa.load(input_path, sr=None, mono=True)
            total_duration = librosa.get_duration(y=audio, sr=sr)

            # Пропускаем файлы короче целевой длительности
            if total_duration < segment_duration:
                print(f"Файл {file_name} слишком короткий ({total_duration:.1f} сек), пропускаем")
                continue

            # Вычисляем количество полных сегментов
            samples_per_segment = int(sr * segment_duration)
            total_segments = int(len(audio) // samples_per_segment)

            # Создаем сегменты без перекрытия
            for i in range(total_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = audio[start:end]

                # Проверка длительности
                segment_duration_actual = librosa.get_duration(y=segment, sr=sr)
                if abs(segment_duration_actual - segment_duration) > 0.1:
                    print(f"Некорректная длительность сегмента {segment_duration_actual:.1f} сек, пропускаем")
                    continue

                output_name = f"{os.path.splitext(file_name)[0]}_part{i + 1:03d}.wav"
                output_path = os.path.join(output_folder, output_name)
                sf.write(output_path, segment, sr)

            print(f"Обработан: {file_name} -> {total_segments} сегментов")

        except Exception as e:
            print(f"Ошибка в файле {file_name}: {str(e)}")


# Пример использования
input_folder = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\z_dataset_bee"
output_folder = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\z_dataset_bee"
split_audio(input_folder, output_folder)