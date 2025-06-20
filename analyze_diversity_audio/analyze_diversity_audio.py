import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import seaborn as sns


def extract_features(file_path, sr=16000):
    """Извлечение признаков из аудиофайла"""
    try:
        y, _ = librosa.load(file_path, sr=sr)

        # Временные характеристики
        zero_crossing = librosa.feature.zero_crossing_rate(y)[0]

        # Спектральные характеристики
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Статистики по признакам
        features = {
            'zcr_mean': np.mean(zero_crossing),
            'zcr_std': np.std(zero_crossing),
            'centroid_mean': np.mean(spectral_centroid),
            'centroid_std': np.std(spectral_centroid),
            'rms_energy': np.sqrt(np.mean(y ** 2))
        }

        # Добавляем MFCC
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def analyze_dataset(directory, output_dir="audio_analysis"):
    """Анализ набора аудиоданных"""
    os.makedirs(output_dir, exist_ok=True)

    # Сбор данных
    features_list = []
    files = [f for f in os.listdir(directory) if f.endswith(('.wav', '.mp3'))]

    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(directory, file)
        features = extract_features(file_path)
        if features:
            features['filename'] = file
            features_list.append(features)

    # Создание DataFrame
    df = pd.DataFrame(features_list).set_index('filename')

    # Сохранение сырых данных
    raw_csv_path = os.path.join(output_dir, "audio_features.csv")
    df.to_csv(raw_csv_path)

    # Визуализация
    plt.figure(figsize=(15, 10))

    # 1. Распределение ключевых признаков
    plt.subplot(2, 2, 1)
    sns.kdeplot(df['zcr_mean'], label='ZCR')
    sns.kdeplot(df['centroid_mean'], label='Spectral Centroid')
    sns.kdeplot(df['rms_energy'], label='RMS Energy')
    plt.title("Feature Distributions")
    plt.legend()

    # 2. PCA для MFCC
    mfcc_cols = [c for c in df.columns if 'mfcc' in c and 'mean' in c]
    X = df[mfcc_cols]

    # Нормализация
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    plt.subplot(2, 2, 2)
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.3)
    plt.xlabel("PC1 (%.2f%%)" % (pca.explained_variance_ratio_[0] * 100))
    plt.ylabel("PC2 (%.2f%%)" % (pca.explained_variance_ratio_[1] * 100))
    plt.title("PCA of MFCC Features")

    # 3. Матрица корреляций
    plt.subplot(2, 2, 3)
    corr_matrix = df[mfcc_cols[:5] + ['zcr_mean', 'centroid_mean', 'rms_energy']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")

    # 4. Статистики
    plt.subplot(2, 2, 4)
    stats = df.describe().loc[['mean', 'std', 'min', 'max']].T
    stats_text = "\n".join([f"{col}: μ={row['mean']:.2f} ± {row['std']:.2f}"
                            for col, row in stats.iterrows()])
    plt.text(0.1, 0.5, stats_text, fontfamily='monospace')
    plt.axis('off')
    plt.title("Key Statistics")

    plt.tight_layout()
    report_path = os.path.join(output_dir, "audio_analysis_report.png")
    plt.savefig(report_path)
    plt.close()

    # Сохраняем PCA модель
    pca_path = os.path.join(output_dir, "pca_model.pkl")
    pd.to_pickle(pca, pca_path)

    return df, report_path


if __name__ == "__main__":
    dataset_path = "D:\\PyCharmProjects\\TinyML_bee_recognition\\dataset\\class_no_bee_3_sec"  # Укажите путь к вашим файлам
    df, report = analyze_dataset(dataset_path)
    print(f"Analysis complete! Report saved to: {report}")