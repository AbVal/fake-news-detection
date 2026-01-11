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
8. Добавить версионирование моделей и данных через DVC
9. Интегрировать MLflow для трекинга экспериментов
10. Упаковать в Docker-образ для оффлайн-инференса

## Установка и запуск

### Требования
```
Python 3.10+ (У автора Python 3.10.2)
CUDA 12 (Если есть желание обучать на GPU)
```

### Установка и запуск

#### Клонирование репозитория
Креды dvc находятся в anytask и необходимы для работы dvc pull, dvc repro и для сборки Docker-образа=ов (при сборке используется скачанная из dvc модель).
Настройка кредов:

```bash
export AWS_ACCESS_KEY_ID='<access_key_id>'
export AWS_SECRET_ACCESS_KEY='<secret_access_key_id>'
```

```bash
git clone https://github.com/AbVal/fake-news-detection.git
cd fake-news-detection
dvc pull
```

#### Установка зависимостей (желательно в виртуальное окружение)
```bash
pip install -r requirements.txt
```

#### Запуск всего пайплайна prepare -> train -> evaluate
```bash
dvc repro
```

#### Сбор данных для обучения и валидации
```bash
python3 prepare_dataset.py --data_path="data" --val_size=0.1 --test_size=0.2 --random_state=42
```

#### Обучение модели
```bash
python3 train.py --config_path="training_params.yaml" --train_data_path="data/train.csv.gz" --val_data_path="data/val.csv.gz"
```
#### Запуск MLflow для мониторинга метрик
```bash
mlflow ui
```
(Авторские логи лежат в папке mlruns и в файле mlflow.db)


#### Сборка образа
При сборке используется cpu-версия pytorch для экономии времени сборки и места, занимаемого контейнером.
```bash
docker build -t ml-app:v1 .
```

#### Запуск оффлайн-инференса
Для запуска оффлайн-инференса можно воспользоваться следующей командой:
```bash
docker run -v ./docker_data:/docker_data ml-app:v1 --input_path /docker_data/input.csv --output_path /docker_data/output.csv
```
В данном случае скрипт ожидает входные данные в формате, считываемом pd.read_csv (csv-файл или сжатые файлы по типу .csv.gz). Данные должны лежать в папке docker_data. Таблица во входных данных должна содержать столбец text -- тексты статей, проверяемых на наличие фейков. Скрипт возвращает данные в csv-файл output.csv в формате таблицы с колонками text (обрезанное начало статьи), prediction (предсказание) и score (уверенность модели).

#### Валидация модели
```bash
python3 validate.py --model_path="model" --data_path="data/test.csv.gz" --metrics_path="metrics.json"
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