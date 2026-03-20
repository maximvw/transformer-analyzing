# Diploma: Transformer State-Tracking in Multi-Step Reasoning

Воспроизведение экспериментов из статьи ["Why Can't Transformers Learn Multiplication?"](pdfs/4x4digits_mul.pdf) (Bai et al., 2025).

Сравниваются два подхода к тренировке трансформера на умножении 4×4 цифр:
- **ICoT** (Implicit Chain-of-Thought) — постепенное удаление CoT-токенов, достигает ~100% accuracy
- **SFT** (Standard Fine-Tuning) — CoT удаляется сразу, <1% accuracy

## Требования

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- GPU с ≥8 GB VRAM (рекомендуется, CPU тоже работает)

## Быстрый старт

```bash
# 1. Генерация данных (80800 train / 1000 val / 1000 test примеров)
make data

# 2. Тренировка ICoT (200 эпох, ~13 эпох до 100% accuracy)
make train-icot

# 3. Тренировка SFT (60 эпох, baseline)
make train-sft
```

Кастомный размер датасета: `make data TRAIN_SIZE=50000`

Кастомная архитектура: `make train-icot N_LAYER=4 N_HEAD=8`

## Resume обучения

Если тренировка прервалась, продолжить с последнего чекпоинта:

```bash
# ICoT resume с checkpoint_5
make resume-icot RESUME_CKPT_ICOT=train_models/4_by_4_mult/icot/checkpoint_5

# SFT resume с checkpoint_10
make resume-sft RESUME_CKPT_SFT=train_models/4_by_4_mult/sft/checkpoint_10
```

Resume восстанавливает: номер шага, текущий `scheduled_to_remove`, состояние оптимайзера, лучшую val accuracy.

## Инференс

```bash
# ICoT
make infer-icot

# SFT
make infer-sft
```

По умолчанию использует `checkpoint_12` (ICoT) и `checkpoint_59` (SFT).
Изменить чекпоинт можно напрямую в `Makefile`.

## Графики из статьи

```bash
make fig2      # Figure 2: Logit Attribution
make fig3      # Figure 3: Linear Probes (c_hat)
make fig5      # Figure 5: Fractals & Minkowski sums
make fig6      # Figure 6: Fourier basis
make fig7      # Figure 7: Gradient norms & losses
make table1    # Table 1: Fourier R² fits

make all-figures  # все сразу
```

Результаты сохраняются в `icot/paper_figures/`.

## Структура проекта

```
diplom/
├── Makefile                          # точка входа для всех команд
├── generate_data.py                  # генерация данных умножения
├── notebooks/train.ipynb             # notebook для Colab/Kaggle
├── Internalize_CoT_Step_by_Step/
│   ├── src/
│   │   ├── train.py                  # основной скрипт обучения
│   │   ├── generate.py               # инференс
│   │   ├── model.py                  # ImplicitModel (GPT-2 based)
│   │   └── data.py                   # CoTDataset
│   └── data/4_by_4_mult/             # train.txt / valid.txt / test.txt
├── train_models/4_by_4_mult/
│   ├── icot/checkpoint_N/            # чекпоинты ICoT
│   └── sft/checkpoint_N/             # чекпоинты SFT
└── icot/experiments/                 # скрипты для графиков статьи
```

## Структура чекпоинта

Каждый `checkpoint_N/` содержит:
| Файл | Содержимое |
|------|-----------|
| `config.json` | конфигурация модели |
| `state_dict.bin` | веса модели |
| `training_state.pt` | шаг, эпоха, `scheduled_to_remove`, `remove_step_counter`, `best_val_accuracy`, состояние оптимайзера |

Чекпоинт сохраняется после каждой эпохи.
