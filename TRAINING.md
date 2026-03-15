# Воспроизведение экспериментов: "Why Can't Transformers Learn Multiplication?"

## Структура проекта

```
diplom/
├── Makefile                           # Все команды запуска
├── generate_data.py                   # Генерация данных
├── pyproject.toml                     # Зависимости (uv)
├── Internalize_CoT_Step_by_Step/      # Тренировка моделей (ICoT, SFT)
│   ├── src/train.py                   # Тренировка (--n_layer, --n_head)
│   ├── src/generate.py                # Инференс
│   └── data/4_by_4_mult/             # Данные (генерируются)
└── icot/                              # Анализ и графики из статьи
    ├── experiments/                   # Скрипты для Figure 2,3,5,6,7 + Table 1
    ├── ckpts/                         # Чекпоинты (кладутся после тренировки)
    └── paper_figures/                 # Выходные PDF-графики
```

## Быстрый старт

```bash
uv sync                # установка зависимостей
make data              # генерация данных (8K train, 1K val, 1K test)
make train-icot        # тренировка ICoT модели (~200 эпох)
make train-sft         # тренировка SFT модели (~60 эпох)
make infer-icot        # проверка accuracy ICoT
make all-figures       # все графики из статьи
```

## Подробнее

### Шаг 0: Зависимости

```bash
uv sync
```

### Шаг 1: Генерация данных

```bash
make data
```

Создаёт 8,000 train / 1,000 valid / 1,000 test примеров в `Internalize_CoT_Step_by_Step/data/4_by_4_mult/`.

### Шаг 2: Тренировка моделей

Все модели — **2 слоя, 4 головы, n_embd=768** (~54M параметров), обучены с нуля.

**ICoT** (постепенное удаление CoT-токенов):
```bash
make train-icot    # ожидаемо 100% accuracy после ~13 эпох
make train-icot N_LAYER=4 N_HEAD=8 N_EMBD=512
```

**SFT** (без CoT — все токены удалены сразу):
```bash
make train-sft     # ожидаемо <1% accuracy (не может выучить средние цифры c3-c6)
```

**Auxiliary Loss** — нужно реализовать самостоятельно, добавив MSE-лосс на линейный probe второго слоя attention (формулы 6-8 из Section 6 статьи). Ожидаемо 99% accuracy.

### Шаг 3: Инференс

```bash
make infer-icot    # checkpoint_12
make infer-sft     # checkpoint_59
```

### Шаг 4: Подготовка чекпоинтов для icot

Скопировать `state_dict.bin` в `icot/ckpts/2L4H/` и создать `config.json` с параметрами модели (n_layer=2, n_head=4, n_embd=768).

### Шаг 5: Графики из статьи

```bash
make fig2      # Figure 2: Logit Attribution heatmap → paper_figures/long_term_effects_heatmap.pdf
make fig3      # Figure 3: Linear Regression Probes → paper_figures/c_hat_probe.pdf
make fig5      # Figure 5: 3D PCA attention heads  → paper_figures/attn_3d_pcas.pdf
make fig6      # Figure 6: Fourier Basis            → paper_figures/fourier_basis.pdf
make fig7      # Figure 7: Gradient norms/losses    → paper_figures/grad_norms_and_losses.pdf
make table1    # Table 1: Fourier R² fits           → вывод в консоль
make all-figures  # всё сразу
```

## Формат данных

Цифры в **обратном порядке** (least significant first): `2365 × 4347` → `5 6 3 2 * 7 4 3 4`

Тренировочный пример с CoT:
```
a0 a1 a2 a3 * b0 b1 b2 b3||<partial_0> + <partial_1> (<running_sum>) + ... + <partial_3> #### c0 c1 c2 c3 c4 c5 c6 c7
```

## Размер моделей и CPU-тренировка

Модель: 2 слоя, 4 головы, n_embd=768 — ~54M параметров (НЕ стандартный GPT-2 124M).

| Платформа | Ориентировочное время |
|---|---|
| M1/M2 Mac (MPS) | ~1-3 часа |
| CPU-only x86 | ~10-20 часов |
| GPU (даже слабый) | ~30-60 минут |

Для экспериментов icot (инференс + графики) CPU достаточно.

## Модификации кода

Изменения в `Internalize_CoT_Step_by_Step`:
1. `src/model.py` — поддержка `gpt2_config` для кастомной архитектуры (from_config вместо from_pretrained)
2. `src/configuration_model.py` — добавлено поле `gpt2_config`
3. `src/train.py` — аргументы `--n_layer`, `--n_head`, `--n_embd`
