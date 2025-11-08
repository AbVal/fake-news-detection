# Детектор фейковых новостей

В современном мире новости появляются быстрее, чем успеваешь их читать. Для того, чтобы понять, настоящая новость или нет, порой необходимо полностью в нее погрузиться.

Этот проект позволяет фильтровать фейковые новости, снижая информационный шум и экономя драгоценное время.

## Цель:
- Создать детектор фейковых новостей
- На вход - текст новости на английском языке, на выход - прогноз фейковости
- Целевые ML-метрики:
    - Recall > 0.9
    - Precision > 0.8
    - F1-Score > 0.8
- Целевые бизнес-метрики:
    - Использование вычислительных ресурсов — в пределах SLA
    - RPS > 500
    - Среднее время отклика сервиса < 5 ms

## Набор данных:
[Fake-News-Detection-dataset](https://huggingface.co/datasets/Pulk17/Fake-News-Detection-dataset)

Формат:
| id: int64 | title: str   | text: str   | subject: str | date: str    | label: int |
|-----------|--------------|-------------|--------------|--------------|------------|
| 1         | sample_title | sample_text | politics     | Nov 25, 2015 | 1          |
| 2         | sample_title | sample_text | world news   | Jan 14, 2019 | 0          |
| 3         | sample_title | sample_text | Middle-east  | Jul 7, 2020  | 1          |

## План экспериментов:
1. Исследовать набор данных
2. Разделить данные на train / validation / test
3. Подобрать архитектуру на основе нейронных сетей (BERT)
4. Обучить модель, измерить ML-метрики
5. Реализовать код с учетом принципов ООП
6. Проверить воспроизводимость экспериментов
7. Покрыть код тестами (unit, integration, проверки типов и pep8)
8. Добавить web-интерфейс и API

## Установка и запуск

### Требования
```
Python 3.10+ (У автора Python 3.10.2)
CUDA 12 (Если есть желание обучать на GPU)
```

### Установка и запуск

#### Клонирование репозитория
```bash
git clone https://github.com/AbVal/fake-news-detection.git
cd fake-news-detection
```

#### Установка зависимостей (желательно в виртуальное окружение)
```bash
pip install -r requirements.txt
```

#### Сбор данных для обучения и валидации
```bash
python3 prepare_dataset.py --data_path="data" --val_size=0.1 --test_size=0.2 --random_state=42
```

#### Обучение модели
```bash
python3 train.py --config_path="training_params.yaml" --train_data_path="data/train.csv.gz" --val_data_path="data/val.csv.gz"
```
#### (Опционально) запуск тензорборда для мониторинга метрик
```bash
tensorboard --logdir model
```
(Авторский файл тензорборда с логами лежит в папке model/runs)


#### Валидация модели
```bash
python3 validate.py --model_path="model" --data_path="data/test.csv.gz"
```
Результат валидации на текущей модели:
| Metric | Value |
|--------|-------|
| Accuracy | 99.71% |
| Precision | 99.58% |
| Recall | 99.86% |
| F1 Score | 99.72% |
| Total Time | 18.98 sec |
| Throughput | 286.51 samples/sec |
| Latency | 3.49 ms |