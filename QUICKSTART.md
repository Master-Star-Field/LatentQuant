# 🚀 Быстрый старт для embd_space_metrics

## ✅ Что было сделано:

Создан модуль **`embd_space_metrics`** для оценки сходства между оригинальными и квантизированными представлениями моделей.

## 📋 Шаги для запуска:

### 1️⃣ Установка (один раз)

```bash
cd /Users/alimalbogaciev/Desktop/yandex_cv_camp/code/LatentQuant
pip install -e .
```

### 2️⃣ Запуск в Jupyter Notebook

```python
# В ячейке Jupyter
import sys
sys.path.insert(0, '/Users/alimalbogaciev/Desktop/yandex_cv_camp/code/LatentQuant')

from embd_space_metrics.evaluation import MetricsEvaluator
from embd_space_metrics.evaluation import visualize_results, save_results, print_results_table

# Настройки
CHECKPOINTS_DIR = './models'
DATA_DIR = './data'
SPLIT = 'trainval'  # 'train', 'test', или 'trainval'
METRICS = ['cka', 'pwcca', 'geometry', 'rsa', 'relational', 'jaccard_knn']

print("🚀 Создание evaluator...")
evaluator = MetricsEvaluator(
    checkpoints_dir=CHECKPOINTS_DIR,
    data_dir=DATA_DIR,
    split=SPLIT,
    metrics=METRICS,
    batch_size=8,
    device='cuda',
    backbone_type='deeplab'
)

print("📊 Запуск оценки...")
results = evaluator.evaluate(verbose=True)

# Результаты
print_results_table(results)
save_results(results, 'evaluation_results.json')
visualize_results(results, output_path='evaluation_plots.png', figsize=(18, 10))

print("✅ Готово!")

# Показать график
from IPython.display import Image
Image('evaluation_plots.png')
```

### 3️⃣ Что получите:

- **`evaluation_results.json`** - результаты всех метрик
- **`evaluation_plots.png`** - визуализация с 6 графиками
- **Таблица в консоли** - отформатированные результаты

## 📊 Доступные метрики:

| Метрика | Описание | Диапазон |
|---------|----------|----------|
| `cka` | Centered Kernel Alignment | 0-1 (выше = лучше) |
| `pwcca` | Projection Weighted CCA | 0-1 (выше = лучше) |
| `geometry` | Geometry Score | -1 до 1 (выше = лучше) |
| `rsa` | Representational Similarity Analysis | -1 до 1 (выше = лучше) |
| `relational` | Relational Knowledge Loss | -1 до 1 (выше = лучше) |
| `jaccard_knn` | Jaccard k-NN сходство | 0-1 (выше = лучше) |

## 🎯 Варианты split:

- **`trainval`** - валидационный датасет (~736 изображений, быстро)
- **`test`** - тестовый датасет (~3669 изображений, полная оценка)
- **`train`** - тренировочный датасет (~2944 изображений)

## 🔧 Быстрый тест (1-2 минуты):

```python
# В Jupyter - только CKA метрика
import sys
sys.path.insert(0, '/Users/alimalbogaciev/Desktop/yandex_cv_camp/code/LatentQuant')

from embd_space_metrics.evaluation import MetricsEvaluator
from embd_space_metrics.evaluation import visualize_results, save_results, print_results_table

evaluator = MetricsEvaluator(
    checkpoints_dir='./models',
    data_dir='./data',
    split='trainval',
    metrics=['cka'],  # Только CKA
    batch_size=8,
    device='cuda'
)

results = evaluator.evaluate(verbose=True)
print_results_table(results)
```

## ⚠️ Важно:

1. **Всегда добавляйте** `sys.path.insert(0, '/Users/alimalbogaciev/Desktop/yandex_cv_camp/code/LatentQuant')` в начале ячейки
2. **Пути к данным** должны быть корректными: `./models` и `./data`
3. **Устройство** `cuda` требует GPU, используйте `cpu` если GPU нет

## 📝 Структура результатов (JSON):

```json
{
  "vq_ema_full_model": {
    "cka": 0.9326,
    "pwcca": 0.6626,
    "geometry": -0.0025,
    "rsa": 0.8775,
    "relational": 0.6947,
    "jaccard_knn": 0.1930
  },
  "fsq_full_model": { ... },
  "lfq_full_model": { ... },
  "residualvq_full_model": { ... }
}
```

## 🐛 Решение проблем:

### Ошибка импорта:
```python
# Проверить путь
import sys
print(sys.path)

# Добавить правильный путь
sys.path.insert(0, '/Users/alimalbogaciev/Desktop/yandex_cv_camp/code/LatentQuant')
```

### CUDA out of memory:
```python
# Уменьшить batch_size
evaluator = MetricsEvaluator(
    batch_size=4,  # вместо 8
    device='cpu'   # или использовать CPU
)
```

## ✅ Проверка установки:

```python
# Проверить что модуль найден
import sys
sys.path.insert(0, '/Users/alimalbogaciev/Desktop/yandex_cv_camp/code/LatentQuant')

import embd_space_metrics
print(f"✅ Модуль найден: {embd_space_metrics.__file__}")

from embd_space_metrics.metrics import list_available_metrics
print(f"✅ Доступные метрики: {list_available_metrics()}")
```

---

**Готовы начать? Скопируйте код из раздела 2️⃣ в Jupyter и запускайте!** 🚀

