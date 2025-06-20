import os
from pydub import AudioSegment


def analyze_audio_files(folder_path):
    # Проверяем, существует ли указанная директория
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не найдена.")
        return

    # Обрабатываем каждый файл в папке
    audio_info = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        try:
            # Загружаем аудиофайл
            audio = AudioSegment.from_file(file_path)

            # Получаем информацию
            info = {
                "Имя файла": file_name,
                "Формат": file_name.split('.')[-1].lower(),
                "Частота дискретизации (Hz)": audio.frame_rate,
                "Число каналов": audio.channels,
                "Длительность (сек)": round(len(audio) / 1000, 2),
                "Битрейт (bps)": audio.frame_rate * audio.sample_width * 8 * audio.channels
            }
            audio_info.append(info)

        except Exception as e:
            print(f"Ошибка при обработке файла {file_name}: {e}")
            continue

    # Вывод информации
    if audio_info:
        print(
            f"{'Имя файла':<25} {'Формат':<10} {'Частота дискр.':<20} {'Каналы':<10} {'Длительность':<15} {'Битрейт':<10}")
        print("-" * 90)
        for info in audio_info:
            print(f"{info['Имя файла']:<25} {info['Формат']:<10} {info['Частота дискретизации (Hz)']:<20} "
                  f"{info['Число каналов']:<10} {info['Длительность (сек)']:<15} {info['Битрейт (bps)']:<10}")
    else:
        print("Аудиофайлы не найдены или произошла ошибка при анализе.")


# Укажите путь к папке с вашими файлами
folder_path = r"D:\PyCharmProjects\TinyML_bee_recognition\main_train\z_dataset_bee"
analyze_audio_files(folder_path)
