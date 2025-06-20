import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def analyze_audio_features(csv_path):
    # Загрузка данных
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Ошибка загрузки файла: {str(e)}"

    # Основные статистики
    try:
        desc_stats = df.drop(columns=['filename']).describe().loc[['mean', 'std', 'min', 'max']].T
    except KeyError:
        return "Файл должен содержать колонку 'filename'"

    # Поиск выбросов (правило 3σ)
    outliers = {}
    for col in df.columns:
        if col == 'filename':
            continue
        try:
            z_scores = np.abs(scipy_stats.zscore(df[col]))
            outliers[col] = np.sum(z_scores > 3)
        except:
            outliers[col] = 'Error'

    # Анализ корреляций
    try:
        numeric_df = df.drop('filename', axis=1)
        corr_matrix = numeric_df.corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        round(corr_matrix.iloc[i, j], 3)
                    ))
    except:
        high_corr = ["Ошибка расчёта корреляций"]

    # PCA анализ
    try:
        X = StandardScaler().fit_transform(numeric_df)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        pca_variance = pca.explained_variance_ratio_
    except:
        pca_variance = [0, 0]
        principal_components = np.zeros((len(df), 2))

    # Формирование отчета
    report = []
    report.append("=" * 50)
    report.append("Анализ аудио-признаков")
    report.append(f"Файл: {csv_path}")
    report.append(f"Всего записей: {len(df)}")
    report.append(f"Колонок: {len(df.columns)}")
    report.append("=" * 50)

    # Основные статистики
    report.append("\nОсновные статистики:")
    for idx, row in desc_stats.iterrows():
        report.append(
            f"{idx:.<25} "
            f"μ={row['mean']:.2f} "
            f"σ={row['std']:.2f} "
            f"[{row['min']:.2f}–{row['max']:.2f}]"
        )

    # Выбросы
    report.append("\n" + "=" * 50)
    report.append("Выбросы (правило 3σ):")
    for col, count in outliers.items():
        report.append(f"{col:.<25} {count}")

    # Корреляции
    report.append("\n" + "=" * 50)
    report.append("Сильные корреляции (>0.8):")
    for corr in high_corr[:10]:  # Ограничиваем 10 самыми сильными
        if isinstance(corr, tuple):
            report.append(f"{corr[0]} ↔ {corr[1]}: {corr[2]}")

    # PCA
    report.append("\n" + "=" * 50)
    report.append("PCA анализ:")
    report.append(f"Объяснённая дисперсия: PC1={pca_variance[0] * 100:.1f}% PC2={pca_variance[1] * 100:.1f}%")
    report.append("Примеры проекций (первые 5):")
    for i in range(5):
        report.append(
            f"{df.filename.iloc[i]:.<30} PC1={principal_components[i, 0]:.2f} PC2={principal_components[i, 1]:.2f}")

    return "\n".join(report)


if __name__ == "__main__":
    csv_path = "D:\\PyCharmProjects\\TinyML_bee_recognition\\analyze_diversity_audio\\audio_analysis\\audio_features.csv"  # Укажите путь к файлу
    analysis_result = analyze_audio_features(csv_path)
    print(analysis_result)