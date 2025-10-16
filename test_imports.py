#!/usr/bin/env python
"""
Тест всех импортов модуля embd_space_metrics
"""

import sys
from pathlib import Path

# Добавить путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("ТЕСТ ИМПОРТОВ МОДУЛЯ embd_space_metrics")
print("="*80)

# Тест 1: Основной модуль
print("\n1. Тест основного модуля...")
try:
    import embd_space_metrics
    print(f"   ✅ embd_space_metrics: {embd_space_metrics.__file__}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

# Тест 2: Метрики
print("\n2. Тест модуля метрик...")
try:
    from embd_space_metrics.metrics import (
        METRIC_REGISTRY,
        create_metric,
        list_available_metrics,
        CKAMetric,
        PWCCAMetric,
        GeometryScoreMetric,
        RSAMetric,
        RelationalKnowledgeLossMetric,
        JaccardKNNMetric
    )
    print(f"   ✅ Все метрики импортированы")
    print(f"   ✅ Доступные метрики: {list_available_metrics()}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# Тест 3: Extraction
print("\n3. Тест модуля extraction...")
try:
    from embd_space_metrics.extraction import (
        extract_features_from_backbone,
        extract_quantized_features,
        load_checkpoint,
        load_original_backbone,
        find_checkpoints
    )
    print(f"   ✅ Все функции extraction импортированы")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# Тест 4: Evaluation
print("\n4. Тест модуля evaluation...")
try:
    from embd_space_metrics.evaluation import (
        MetricsEvaluator,
        visualize_results,
        save_results,
        print_results_table
    )
    print(f"   ✅ Все функции evaluation импортированы")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# Тест 5: CLI
print("\n5. Тест CLI модуля...")
try:
    from embd_space_metrics import cli
    print(f"   ✅ CLI модуль импортирован")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# Тест 6: Создание метрики
print("\n6. Тест создания метрики...")
try:
    import torch
    metric = create_metric('cka', device='cpu')
    dummy_features = torch.randn(10, 512)
    score = metric.compute(dummy_features, dummy_features)
    print(f"   ✅ CKA метрика создана и протестирована: {score:.4f}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ТЕСТ ЗАВЕРШЁН")
print("="*80)

