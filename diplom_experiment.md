# Дизайн экспериментов для диплома: Graph Connectivity с DSU Auxiliary Loss

## 0. Общий замысел

Цель диплома — показать, что auxiliary state-tracking loss (аналог running sum probe из статьи Bai et al.) **переносится** с задачи умножения на задачу graph reachability. Вместо running sum используется массив parent[] из алгоритма Disjoint Set Union (DSU), который обновляется при обработке каждого ребра.

Гипотеза: SFT-модель выучит поверхностные эвристики (степень вершин, локальная связность) и провалится на OOD-графах, а модель с DSU auxiliary loss **интернализирует** алгоритм и будет генерализовать.

---

## 1. Задача и формат данных

### 1.1 Задача

**Graph Connectivity (Reachability):** Дан неориентированный граф в виде последовательности рёбер и запрос `Query(u, v)`. Нужно ответить `1` (достижимы) или `0` (не достижимы).

### 1.2 Формат входной последовательности

```
<START> E(u1,v1) E(u2,v2) ... E(um,vm) <SEP> Q(u,v) <ANS> {0|1} <END>
```

**Пример (граф на 4 вершинах):**
```
<START> E(0,1) E(2,3) E(1,2) <SEP> Q(0,3) <ANS> 1 <END>
```

**Токенизация:** Каждый элемент — отдельный токен. Словарь:
- Специальные: `<START>`, `<SEP>`, `<ANS>`, `<END>`, `<PAD>`
- Рёбра: `E(i,j)` для всех допустимых пар i,j (или раздельные токены `E(`, `i`, `,`, `j`, `)`)
- Запрос: `Q(i,j)` аналогично
- Ответ: `0`, `1`

**Решение о токенизации (нужно выбрать):**

**Вариант A — Compound tokens:** Каждое ребро `E(i,j)` — один токен. Словарь: O(N^2) токенов для рёбер + O(N^2) для запросов + спец. токены. Простая обработка, но не масштабируется для больших N.

**Вариант B — Factored tokens:** Раздельная токенизация: `E`, `(`, `0`, `,`, `1`, `)`. Словарь фиксированного размера O(N + const). Более длинные последовательности, но масштабируется.

**Рекомендация:** Для N ≤ 50 использовать **Вариант A** (compound tokens) — это ближе к оригинальной статье, где каждая цифра = один токен, и упрощает анализ attention patterns. Размер словаря ~2500 + спец. токены — вполне допустимо.

### 1.3 Порядок рёбер

Рёбра подаются в **случайном** порядке (shuffle при генерации). Это важно: модель не должна полагаться на порядок рёбер.

### 1.4 Каноникализация рёбер

Для каждого ребра `(u,v)`: всегда записываем `E(min(u,v), max(u,v))`, чтобы избежать дублирования.

---

## 2. Архитектура модели

### 2.1 Базовая модель

| Параметр | Значение | Обоснование |
|---|---|---|
| Архитектура | GPT-2 (decoder-only) | Как в оригинальной статье |
| Число слоёв | **4** | Графы сложнее умножения; 2 слоя может не хватить для path длины > 2 |
| Число голов внимания | 4 | Как в оригинале |
| d_model | 256 | Как в оригинале |
| Обучение | С нуля | Без предобучения, чтобы исключить confounding |

**Почему 4 слоя?** Для graph connectivity трансформер должен делать "message passing" через attention — каждый слой может распространить связность на 1 hop. Для графов диаметра d нужно ~d слоёв. С 4 слоями модель теоретически может решать графы до диаметра 4 без auxiliary loss (но на практике не будет из-за local optima).

### 2.2 Probe для auxiliary loss

Линейный regression head крепится к **выходам attention heads последнего слоя** (до MLP), аналогично оригинальной статье.

- Для каждой головы h: `z^h_{t} = w_h^T * ATT_h^L(·)`
- `w_h ∈ R^{d_head}` — обучаемый вектор (не замороженный, обучается совместно с моделью)

---

## 3. DSU Auxiliary Loss

### 3.1 Алгоритмическое состояние (таргет для probe)

Для графа с N вершинами определяем массив `parent[0..N-1]`:

**Начальное состояние** (до обработки рёбер):
```
parent = [0, 1, 2, ..., N-1]    (каждая вершина — свой компонент)
```

**При обработке ребра E(u,v):**
```
root_u = Find(u)   // с path compression
root_v = Find(v)
if root_u != root_v:
    parent[root_v] = root_u   // Union (без rank, для простоты)
```

**Упрощение для предсказуемости:** Используем Union **без** rank/size — всегда `parent[root_v] = root_u` (merge в сторону меньшего корня). Это делает таргет детерминированным для данного порядка рёбер.

**Альтернатива (component ID):** Вместо parent[] можно использовать `comp[i] = Find(i)` — массив component IDs после каждого ребра. Это более "информативный" таргет, т.к. напрямую показывает, какие вершины в одном компоненте.

**Рекомендация:** Использовать **component ID** массив `comp[i] = Find(i)` как таргет. Это:
- Прямее связано с финальной задачей (reachability = сравнение comp[u] и comp[v])
- Проще для probe (не нужно восстанавливать транзитивное замыкание)
- Аналогично running sum в оригинале: одно число на каждый timestep, суммаризующее всё вычисление до этого момента

### 3.2 Формат таргета

На позиции каждого ребра `E(u_t, v_t)` (timestep t) таргет для probe:
```
comp_t = [Find(0), Find(1), ..., Find(N-1)]    после обработки ребра t
```

Это вектор из N целых чисел (component IDs).

### 3.3 Формулировка loss

**State-tracking loss (cross-entropy):**
```
L_state = (1/H) * sum_{h=1}^{H} (1/M) * sum_{t=1}^{M} (1/N) * sum_{i=0}^{N-1} CE(probe_h(ATT_h^L, t, i), comp_t[i])
```
где:
- M — число рёбер в последовательности
- N — число вершин
- `probe_h(ATT_h^L, t, i)` — предсказание probe для компоненты вершины i на шаге t от head h
- `comp_t[i]` — истинный component ID вершины i после обработки t-го ребра
- CE — cross-entropy (component ID — дискретная переменная ∈ {0, ..., N-1})

**Probe architecture:**
```
logits_{t,i}^h = W_h * ATT_h^L(t) + b_h    // W_h ∈ R^{N x d_head}, b_h ∈ R^N
```
Отдельный linear layer для предсказания comp[i] по attention output на позиции t. Предсказывает **N-way classification** для каждой из N вершин.

**Полный loss:**
```
L = L_LM + lambda * L_state
```
где `L_LM` — cross-entropy на финальный ответ (0 или 1).

### 3.4 Важное отличие от оригинала

| | Умножение (оригинал) | Graph Connectivity (наш) |
|---|---|---|
| Таргет probe | Скаляр c_hat_k (running sum) | Вектор comp[0..N-1] (component IDs) |
| Loss probe | MSE (регрессия) | Cross-Entropy (классификация) |
| Размерность таргета | 1 число на timestep | N чисел на timestep |
| Где крепится | Layer 2 attention out | Layer L (последний) attention out |
| Probe | Вектор w_h | Матрица W_h ∈ R^{N x d_head} |

---

## 4. Генерация данных

### 4.1 Типы графов (train)

| Тип | Параметры | Доля в train set |
|---|---|---|
| Erdos-Renyi G(N, p) | p ∈ {0.05, 0.10, 0.15, 0.20, 0.30} | 50% |
| Random trees | N-1 рёбер, случайное дерево (Prufer sequence) | 20% |
| Sparse random | m = N * k рёбер, k ∈ {1.0, 1.5, 2.0} | 20% |
| Complete graph (малые N) | K_n, n ∈ {3,4,5} | 5% |
| Path graph | Цепочка 0-1-2-..-(N-1) | 5% |

### 4.2 Размеры графов

- **Train:** N ∈ {8, 10, 12, 15, 20}
- **Validation (in-distribution):** Те же N, те же типы, другие random seeds
- **Test (in-distribution):** Те же N, те же типы, другие random seeds

### 4.3 OOD Test sets (ключевое для демонстрации генерализации)

| OOD Test Set | Описание | Цель |
|---|---|---|
| **Large N** | N ∈ {25, 30, 40, 50} | Генерализация по размеру |
| **Cyclic Grids** | Решётчатые графы (grid), cycles | Типология, не встречавшаяся в train |
| **Adversarial Degree** | Графы, где высокая степень ≠ связность (hub + isolated components) | Сломать degree-based heuristic |
| **Long Diameter** | Графы с диаметром >> 4 (длинные цепочки + шум) | Сломать shallow message passing |
| **Disconnected Dense** | Два плотных компонента без связи между ними | Сломать density-based heuristic |

### 4.4 Баланс классов

Примерно 50/50 по ответу (reachable / not reachable). Для каждого графа генерируем:
- Запросы (u,v) где u и v в одном компоненте → label 1
- Запросы (u,v) где u и v в разных компонентах → label 0

### 4.5 Размеры датасетов

| Split | Размер |
|---|---|
| Train | 80,000 примеров |
| Validation | 2,000 |
| Test (in-distribution) | 2,000 |
| Test (OOD, каждый) | 1,000 |

---

## 5. Стадии экспериментов

### Стадия 1: Baseline SFT

**Цель:** Показать, что стандартный SFT застревает в локальном оптимуме / выучивает эвристики.

**Настройка:**
- Loss: только `L_LM` (cross-entropy на ответ)
- Без CoT, без auxiliary loss
- Архитектура: 4L4H GPT-2
- LR: 5e-5, AdamW, weight decay 0.01
- Batch size: 32
- Тренировать до сходимости (≤ 50 эпох)

**Ожидания:**
- ID accuracy: 70-90% (модель выучит эвристики)
- OOD accuracy: ~50% (случайный уровень) на adversarial и large-diameter тестах
- Модель будет коррелировать предсказания со степенью вершин

**Что мерить:**
- Train/Val loss curves
- Exact match accuracy (per-sample)
- Accuracy breakdown по типам графов и по N
- OOD accuracy на каждом adversarial test set
- Корреляция предсказания с node degree

### Стадия 2: Auxiliary Loss (DSU state-tracking)

**Цель:** Главный эксперимент — показать, что DSU auxiliary loss даёт 99-100% accuracy без CoT.

**Настройка:**
- Loss: `L = L_LM + lambda * L_state`
- Probe: линейный, крепится к attention outputs последнего слоя
- lambda: подбирать из {0.1, 0.5, 1.0, 2.0, 5.0}
- Всё остальное как в Стадии 1

**Ожидания:**
- ID accuracy: 99-100%
- OOD accuracy: значительно выше SFT
- Learning dynamics: модель выучивает "лёгкие" компоненты (маленькие графы, высокая плотность) первыми, затем — сложные

**Что мерить:**
- Train/Val loss curves (L_LM и L_state отдельно)
- Accuracy по эпохам
- OOD accuracy
- Per-edge state-tracking accuracy (может ли probe предсказать DSU state)
- **Ablation по lambda:** как accuracy зависит от веса auxiliary loss

### Стадия 3: Ablation Studies

**3a. Без auxiliary loss (= Стадия 1)** — уже есть.

**3b. Только auxiliary loss (lambda → ∞):**
- L = L_state (без L_LM)
- Показать, что без основного task loss модель не учится отвечать на query

**3c. Размер модели:**
- 2L4H (как в оригинале): ожидается хуже, чем 4L4H
- 6L4H: ожидается не сильно лучше 4L4H (с auxiliary loss масштаб не так важен)
- 4L8H: проверить влияние числа голов

**3d. Глубокая модель без auxiliary loss:**
- SFT с 6L4H или 8L4H — проверить, "спасёт ли" глубина

**3e. Random auxiliary target (контроль):**
- Probe предсказывает случайный вектор вместо истинного DSU state
- Должна показать, что дело не в "дополнительном градиенте", а именно в DSU-aligned supervision

### Стадия 4: Анализ внутренних механизмов

**Цель:** Mechanistic interpretability — понять, **как** модель решает задачу.

**4a. Logit Attribution (аналог Figure 2 из оригинала):**
- Для каждого ребра E(u,v) во входе: заменить на случайное E(u',v') и измерить изменение логита ответа
- Показать, что модель с auxiliary loss обращает внимание на рёбра, лежащие на пути u→v, а SFT — нет

**4b. Linear Probing (аналог Figure 3):**
- На hidden states каждого слоя обучить post-hoc probe, предсказывающий DSU state
- Сравнить probe accuracy между SFT и Auxiliary Loss моделями
- Ожидание: Aux Loss модель >> SFT по probe accuracy

**4c. Attention Patterns (аналог Figure 4, 8, 10):**
- Визуализировать attention maps для всех heads на конкретных примерах
- Ожидание для Aux Loss модели:
  - Layer 1 heads: обращают внимание на отдельные рёбра, кэшируя связность
  - Deeper layers: агрегируют — формируют "цепочки" (path-like patterns)
  - На позиции Query: attention обращается к рёбрам, образующим путь между query вершинами

**4d. Feature Geometry (аналог Figure 5, 6):**
- 3D PCA скрытых состояний на позиции Query token
- Ожидание: вершины из одного компонента кластеризуются вместе
- Возможно, обнаружатся структуры, аналогичные Fourier basis (но для дискретных компонентов, скорее one-hot-like)

**4e. Per-edge learning dynamics (аналог Figure 7):**
- Loss и gradient norms **per edge position** по ходу обучения
- Для SFT: ожидается, что loss для queries с длинными путями не падает (аналог "средних цифр" в умножении)
- Для Aux Loss: loss падает для всех query типов

---

## 6. Метрики

| Метрика | Описание |
|---|---|
| **Accuracy (exact match)** | Доля правильных ответов 0/1 |
| **ID Accuracy** | Accuracy на in-distribution test set |
| **OOD Accuracy** | Accuracy на каждом OOD test set отдельно |
| **State-tracking MAE** | Mean Absolute Error probe предсказаний vs истинный comp[] |
| **State-tracking Accuracy** | Доля правильно предсказанных component IDs |
| **Degree Correlation** | Корреляция Спирмена между degree вершины и P(reachable) |
| **Per-diameter Accuracy** | Accuracy в зависимости от расстояния между query вершинами |
| **Loss curves** | L_LM, L_state, L_total по шагам |

---

## 7. Гиперпараметры (сводка)

| Параметр | Значение |
|---|---|
| Архитектура | 4-layer, 4-head GPT-2, d_model=256 |
| Optimizer | AdamW |
| LR | 5e-5 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Max sequence length | ~200 tokens (для N=20, m≈30 edges) |
| Train samples | 80,000 |
| Val/Test samples | 2,000 / 2,000 |
| lambda (aux loss weight) | Grid search: {0.1, 0.5, 1.0, 2.0, 5.0} |
| Training epochs | До 50 (с early stopping по val loss) |
| GPU | 1x (A100/H100/что доступно) |

---

## 8. Ожидаемые результаты (сводная таблица)

| Модель | ID Acc | OOD Large N | OOD Adversarial | OOD Long Diam |
|---|---|---|---|---|
| SFT (4L4H) | 70-90% | 50-60% | ~50% | ~50% |
| **Aux Loss (4L4H)** | **99-100%** | **85-99%** | **80-95%** | **75-90%** |
| Random Aux (control) | 70-90% | 50-60% | ~50% | ~50% |

---

## 9. План реализации (порядок)

1. **Генерация данных** — скрипт `generate_data.py` (уже начат)
   - Генерация графов всех типов
   - Вычисление DSU states для каждого шага
   - Сохранение в формате для тренировки

2. **Модель + SFT training** — базовый пайплайн
   - GPT-2 from scratch
   - Dataloader для graph sequences
   - Training loop с логированием per-step

3. **Auxiliary loss** — добавить probe heads
   - Linear probes к attention outputs
   - DSU state cross-entropy loss
   - Lambda scheduling (опционально)

4. **Evaluation pipeline** — автоматический запуск всех тестов
   - ID/OOD accuracy
   - State-tracking metrics
   - Attention visualization

5. **Analysis & Interpretability** — logit attribution, probing, attention patterns
   - Jupyter notebooks для визуализации
   - Сравнение механизмов SFT vs Aux Loss

---

## 10. Риски и fallback-планы

| Риск | Вероятность | Mitigation |
|---|---|---|
| 4L4H не хватает для графов N=20 | Средняя | Увеличить до 6L4H; уменьшить N до 10-15 |
| DSU state слишком сложен для linear probe | Средняя | Использовать MLP probe (2 слоя) вместо linear |
| SFT работает слишком хорошо (нет проблемы) | Низкая | Увеличить N или усложнить OOD tests |
| Compute ограничения | Средняя | Уменьшить N, уменьшить train set, использовать fp16 |
| Aux Loss не помогает | Низкая | Проверить: правильный ли таргет? MLP probe? Другой слой? |
