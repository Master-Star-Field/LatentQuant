# Embedding Space Metrics Evaluation

Модуль для оценки сходства между оригинальными и квантизированными представлениями моделей сегментации с использованием различных метрик.

## Возможности

- **6 метрик сходства**: CKA, PWCCA, Geometry Score, RSA, Relational Knowledge Loss, Jaccard k-NN
- **Поддержка нескольких бэкбонов**: DeepLabV3-ResNet50, ViT
- **Поддержка различных квантайзеров**: VQ-EMA, FSQ, LFQ, ResidualVQ
- **Гибкий CLI**: Простой интерфейс командной строки с полной документацией
- **Автоматическое определение**: Автоматическое определение типа модели из чекпоинтов
- **Эффективность по памяти**: Spatial pooling для работы на больших датасетах

## Установка

Модуль уже включен в проект LatentQuant. Убедитесь, что установлены все зависимости:

```bash
cd LatentQuant
pip install -e .
```

## Быстрый старт

### Командная строка

```bash
# Показать справку
python -m embd_space_metrics.cli --help

# Список доступных метрик
python -m embd_space_metrics.cli list-metrics

# Оценить все модели со всеми метриками
python -m embd_space_metrics.cli evaluate \
    --checkpoints-dir ./models \
    --split trainval \
    --output results.json

# Оценить с конкретными метриками
python -m embd_space_metrics.cli evaluate \
    --checkpoints-dir ./models \
    --metrics cka pwcca rsa \
    --split test \
    --output test_results.json \
    --visualize test_plots.png

# Визуализировать сохраненные результаты
python -m embd_space_metrics.cli visualize \
    --results results.json \
    --output plots.png
```

### Python API

```python
from embd_space_metrics.evaluation import MetricsEvaluator
from embd_space_metrics.evaluation import visualize_results, save_results

# Создать evaluator
evaluator = MetricsEvaluator(
    checkpoints_dir='./models',
    data_dir='./data',
    split='trainval',
    metrics=['cka', 'pwcca', 'rsa'],
    batch_size=8,
    device='cuda'
)

# Запустить оценку
results = evaluator.evaluate()

# Сохранить результаты
save_results(results, 'results.json')

# Визуализировать
visualize_results(results, output_path='plots.png')
```

## Доступные метрики

| Метрика | Описание | Диапазон |
|---------|----------|----------|
| `cka` | Centered Kernel Alignment | 0-1 (выше = лучше) |
| `pwcca` | Projection Weighted CCA | 0-1 (выше = лучше) |
| `geometry` | Geometry Score (Spearman корреляция расстояний) | -1-1 (выше = лучше) |
| `rsa` | Representational Similarity Analysis | -1-1 (выше = лучше) |
| `relational` | Relational Knowledge Loss (Pearson корреляция) | -1-1 (выше = лучше) |
| `jaccard_knn` | Jaccard k-NN сходство | 0-1 (выше = лучше) |

## CLI Команды

### `evaluate` - Оценка моделей

```bash
python -m embd_space_metrics.cli evaluate --help
```

**Основные аргументы:**
- `--checkpoints-dir`: Директория с чекпоинтами (обязательно)
- `--data-dir`: Путь к датасету (по умолчанию: `./data`)
- `--split`: Раздел датасета (`train`, `test`, `trainval`)
- `--metrics`: Список метрик для вычисления
- `--backbone`: Тип бэкбона (`deeplab` или `vit`)
- `--batch-size`: Размер батча (по умолчанию: 8)
- `--output`: Путь для сохранения JSON результатов
- `--visualize`: Путь для сохранения визуализации
- `--device`: Устройство (`cuda` или `cpu`)

### `list-metrics` - Список метрик

```bash
python -m embd_space_metrics.cli list-metrics
```

### `visualize` - Визуализация результатов

```bash
python -m embd_space_metrics.cli visualize \
    --results results.json \
    --output plots.png \
    --figsize 12 6
```

## Поддерживаемые форматы чекпоинтов

Модуль поддерживает два формата:

1. **PyTorch Lightning чекпоинты** (`.ckpt`):
   - Автоматическое определение типа модели
   - Загрузка из `state_dict` и `hyper_parameters`

2. **Raw PyTorch чекпоинты** (`.pth`):
   - Формат из ноутбука `model_testing_v2.ipynb`
   - Содержит `encoder_state_dict`, `decoder_state_dict`, `quantizer_state_dict`

## Примеры использования

### 1. Оценка DeepLabV3 моделей на тестовом датасете

```bash
python -m embd_space_metrics.cli evaluate \
    --checkpoints-dir ./models \
    --split test \
    --backbone deeplab \
    --metrics cka pwcca geometry rsa relational jaccard_knn \
    --output deeplab_test_results.json \
    --visualize deeplab_test_plots.png
```

### 2. Оценка ViT моделей на trainval

```bash
python -m embd_space_metrics.cli evaluate \
    --checkpoints-dir ./lightning_logs/vit_experiments \
    --split trainval \
    --backbone vit \
    --output vit_results.json
```

### 3. Быстрая оценка одной метрики

```bash
python -m embd_space_metrics.cli evaluate \
    --checkpoints-dir ./models \
    --metrics cka \
    --split train \
    --output cka_only.json \
    --quiet
```

## Структура модуля

```
embd_space_metrics/
├── __init__.py                 # Главный модуль
├── cli.py                      # CLI интерфейс
├── README.md                   # Документация
├── metrics/                    # Метрики сходства
│   ├── __init__.py
│   ├── base.py                # Базовый класс метрики
│   ├── cka.py                 # CKA метрика
│   ├── pwcca.py               # PWCCA метрика
│   ├── geometry.py            # Geometry Score
│   ├── rsa.py                 # RSA метрика
│   ├── relational.py          # Relational Knowledge Loss
│   ├── jaccard.py             # Jaccard k-NN
│   ├── helpers.py             # Вспомогательные функции
│   └── registry.py            # Реестр метрик
├── extraction/                 # Извлечение признаков
│   ├── __init__.py
│   ├── features.py            # Функции извлечения
│   └── model_loader.py        # Загрузка моделей
├── evaluation/                 # Оценка и визуализация
│   ├── __init__.py
│   ├── evaluator.py           # Основной evaluator
│   └── visualizer.py          # Визуализация результатов
└── model_testing_v2.ipynb     # Оригинальный ноутбук
```

## Интеграция с существующим кодом

Модуль использует существующую инфраструктуру из `embeddings_squeeze`:

- **Бэкбоны**: `DeepLabV3SegmentationBackbone`, `ViTSegmentationBackbone`
- **Квантайзеры**: `VQWithProjection`, `FSQWithProjection`, `LFQWithProjection`, `ResidualVQWithProjection`
- **Данные**: `OxfordPetDataModule`

Никакие изменения в `embeddings_squeeze/` не требуются.

## Результаты

Результаты сохраняются в JSON формате:

```json
{
  "vq_ema_full_model": {
    "cka": 0.9729,
    "pwcca": 0.8945,
    "geometry": 0.7832,
    "rsa": 0.8123,
    "relational": 0.8234,
    "jaccard_knn": 0.7654
  },
  "fsq_full_model": {
    ...
  }
}
```

## Решение проблем

### Out of Memory ошибки

Модуль использует spatial pooling для эффективности по памяти. Если возникают OOM ошибки:

- Уменьшите `--batch-size`
- Используйте `--split train` вместо `trainval` для меньшего датасета
- Убедитесь, что используется GPU для вычислений

### Checkpoint не найден

Убедитесь, что:
- Директория `--checkpoints-dir` содержит файлы `.pth` или `.ckpt`
- Путь указан правильно (абсолютный или относительный)

### Неизвестная метрика

Проверьте список доступных метрик:
```bash
python -m embd_space_metrics.cli list-metrics
```

## Ссылки

- **CKA**: Kornblith et al. "Similarity of Neural Network Representations Revisited" (ICML 2019)
- **PWCCA**: Morcos et al. "Insights on representational similarity in neural networks with canonical correlation" (NeurIPS 2018)
- **Geometry Score**: Shahbazi et al. "Geometry Score" (NeurIPS 2021)
- **RSA**: Kriegeskorte et al. "Representational similarity analysis" (2008)
- **Relational Knowledge**: Park et al. "Relational Knowledge Distillation" (CVPR 2019)

## Лицензия

MIT License (см. корневой LICENSE файл проекта)

