import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sweetviz as sv


def main():
    df_train = pd.read_csv("train.csv")
    display_cat_features(df_train)
    get_report(df_train)
    display_cat_features(df_train)


# распределение категориальные признаки
def display_cat_features(data: pd.DataFrame):
    df_train = data.select_dtypes("object")
    fig, axes = plt.subplots(round(len(df_train.columns) / 3), 3, figsize=(15, 30))

    for i, ax in enumerate(fig.axes):
        if i < len(df_train):
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
            sns.countplot(x=df_train.columns[i], alpha=0.7, data=df_train, ax=ax)

    fig.tight_layout()


# Получаем отсчет
def get_report(data: pd.DataFrame):
    my_report = sv.analyze(data)
    my_report.show_html()


# Отображаем матрицу корреляций
def display_corr(data: pd.DataFrame):
    num_features = data.select_dtypes(exclude="object")
    correlation_matrix = num_features.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, fmt="0.2f", cmap="YlGnBu", annot=True)
    plt.tight_layout()
    plt.show()


# Отображаем NaN значения
def display_nan(data):
    nan_counts = data.isna().sum()

    # Построение столбчатой диаграммы
    plt.figure(figsize=(12, 6))
    nan_counts.plot(kind="bar", color="skyblue")
    plt.title("Количество NaN, где есть пропуски)")
    plt.xlabel("Столбцы")
    plt.ylabel("Количество NaN")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


# Отображаем сколько нулей там, где есть нули
def display_zero(data):
    zero_counts = (data == 0).sum()
    zero_counts = zero_counts[zero_counts > 0]
    plt.figure(figsize=(12, 6))
    zero_counts.plot(kind="bar", color="skyblue")
    plt.title("Количество значений, равных 0, где есть 0")
    plt.xlabel("Столбцы")
    plt.ylabel("Количество значений, равных 0")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


# Отображаем распределение целевого признака
def display_target(data: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data["SalePrice"], kde=True, color="skyblue"
    )  # Гистограмма с ядерной оценкой плотности, используется для визуализации формы распределения данных и предоставляет непараметрическую оценку плотности
    plt.title("Распределение Sale Price")
    plt.xlabel("Sale Price")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
