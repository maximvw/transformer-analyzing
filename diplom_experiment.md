# Дизайн экспериментов для диплома: Graph Connectivity с DSU Auxiliary Loss

## 0. Общий замысел

Цель диплома — показать, что auxiliary state-tracking loss (аналог running sum probe из статьи Bai et al.) **переносится** с задачи умножения на задачу graph reachability. Вместо running sum используется массив comp[] из алгоритма Disjoint Set Union (DSU), который обновляется при обработке каждого ребра.

Гипотеза: SFT-модель выучит поверхностные эвристики (степень вершин, локальная связность) и провалится на OOD-графах, а модель с DSU auxiliary loss **интернализирует** алгоритм и будет генерализовать.

---

## 1. Задача и формат данных

### 1.1 Задача

**Graph Connectivity (Reachability):** Дан неориентированный граф в виде последовательности рёбер и запрос `Query(u, v)`. Нужно ответить `1` (достижимы) или `0` (не достижимы).

### 1.2 Формат входной последовательности

```
<START> E(u1,v1) E(u2,v2) ... E(um,vm) <SEP> Q(u,v) <ANS> {0|1} <END>
```

**Пример (граф на 5 вершинах):**
```
<START> E(0,1) E(2,3) E(1,2) E(3,4) <SEP> Q(0,3) <ANS> 1 <END>
```

### 1.3 Токенизация

**Compound tokens:** Каждое ребро `E(i,j)` — один токен, каждый запрос `Q(i,j)` — один токен. Словарь:
- Специальные: `<START>`, `<SEP>`, `<ANS>`, `<END>`, `<PAD>`
- Рёбра: `E(i,j)` для всех допустимых пар (i < j)
- Запросы: `Q(i,j)` для всех пар
- Ответ: `0`, `1`

Для max_N = 30: ~435 edge tokens + ~870 query tokens + спец. = ~1310 токенов. Компактно и ближе к оригинальной статье, где каждая цифра = один токен.

### 1.4 Каноникализация рёбер

Для каждого ребра `(u,v)`: всегда `E(min(u,v), max(u,v))`.

### 1.5 Порядок рёбер

Рёбра подаются в **случайном** порядке. Shuffle выполняется **на лету** в DataLoader (каждая эпоха — новый порядок для того же графа). Это даёт бесконечную аугментацию без увеличения датасета.

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

Линейный classification head крепится к **выходам attention heads последнего слоя** (до MLP), аналогично оригинальной статье.

Probe предсказывает `comp[i]` (component ID вершины i) для каждой вершины на каждой позиции ребра.

**Архитектура probe (для каждой головы h):**
```
logits_{t}^h = W_h * ATT_h^L(t) + b_h    // W_h ∈ R^{(max_N·max_N) × d_head}, b_h ∈ R^{max_N·max_N}
reshape → [max_N, max_N]                   // [vertex_idx, class_logits]
```

- Строка `i` результата = `max_N` логитов для N-way classification: предсказание `comp[i]` ∈ {0, ..., max_N-1}
- `d_head = d_model / H = 256 / 4 = 64`
- Probe обучается **совместно** с основной моделью (не заморожен), градиенты от `L_state` проходят через probe в attention heads

---

## 3. DSU Auxiliary Loss

### 3.1 Алгоритмическое состояние (таргет для probe)

**Component ID:** `comp[i] = min вершина в компоненте вершины i` — каноническая нумерация.

**Union by min:** При `Union(u, v)` корень объединённого компонента = `min(root_u, root_v)`. Это гарантирует, что `comp[i]` всегда равен минимальной вершине в компоненте, что делает таргет **детерминированным** — результат зависит только от множества обработанных рёбер на данный момент, но **не** от порядка внутри них (на каждом промежуточном шаге `comp[]` определяется однозначно для данного набора рёбер до этого шага).

Пример (N=5, рёбра в порядке E(0,1), E(2,3), E(1,2), E(3,4)):
```
начало:       comp = [0, 1, 2, 3, 4]
после E(0,1): comp = [0, 0, 2, 3, 4]
после E(2,3): comp = [0, 0, 2, 2, 4]
после E(1,2): comp = [0, 0, 0, 0, 4]   ← merge {0,1} и {2,3}
после E(3,4): comp = [0, 0, 0, 0, 0]
```

**Свойства таргета:**
- **Богатый сигнал:** N значений на каждом шаге (полная информация о связности)
- **Точный аналог running sum:** полностью определяет state и ответ на любой query
- **Порядко-зависимый** на промежуточных шагах (это нормально — модель обрабатывает рёбра последовательно, как цифры в умножении)

### 3.2 Переменное N: padding + masking

Для работы с графами разного размера одной моделью:
- Фиксируем `max_N = 30`
- Probe всегда выдаёт `max_N × max_N` логитов
- Loss считается **только по первым N вершинам**, остальные замаскированы

```
Граф N=10: comp = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, -, -, ..., -]
                                                     ^^^^^^^^^^^
                                                     замаскировано
```

### 3.3 Формулировка loss

**State-tracking loss (cross-entropy):**
```
L_state = (1/H) * sum_h (1/M) * sum_t (1/N) * sum_{i=0}^{N-1} CE(logits_{t,i}^h, comp_t[i])
```

**Полный loss:**
```
L = L_LM + lambda * L_state
```
где `L_LM` — **classification loss**: отдельный classification head `Linear(d_model, 2)` применяется к hidden state на позиции токена `<ANS>`, результат сравнивается с target (0 или 1) через cross-entropy. Это **не** стандартный autoregressive LM loss, а classification head — модель обучается отвечать на query через бинарную классификацию.

### 3.4 Важное отличие от оригинала

| | Умножение (оригинал) | Graph Connectivity (наш) |
|---|---|---|
| Таргет probe | Скаляр c_hat_k (running sum) | Вектор comp[0..N-1] (component IDs) |
| Loss probe | MSE (регрессия) | Cross-Entropy (классификация) |
| Размерность таргета | 1 число на timestep | N чисел на timestep (padded до max_N) |
| Где крепится | Layer 2 attention out | Layer L (последний) attention out |
| Probe | Вектор w_h ∈ R^{d_head} | Матрица W_h ∈ R^{(max_N·max_N) × d_head} |
| Переменный размер входа | Нет (всегда 4x4) | Да (N от 8 до 30, padding + masking) |

---

## 4. Генерация данных и DataLoader

### 4.1 Хранение на диске

На диске хранятся только **графы** (edge lists + N). DSU states и shuffle вычисляются на лету.

```python
# Файл: graphs_train.json
[
    {"n": 10, "edges": [[0,1], [2,3], [1,2], ...]},
    {"n": 15, "edges": [[0,4], [3,7], ...]},
    ...
]
```

### 4.2 DataLoader (on-the-fly augmentation)

При каждом обращении к примеру:
1. Случайный shuffle рёбер
2. Случайный выбор query (u, v) + вычисление label
3. Вычисление comp[] states через DSU (микросекунды)

```python
def __getitem__(self, idx):
    graph = self.graphs[idx]
    N, edges = graph['n'], graph['edges']

    # 1. Новый порядок рёбер каждый раз
    perm = torch.randperm(len(edges))
    shuffled = [edges[i] for i in perm]

    # 2. Случайный query + label
    u, v = random.sample(range(N), 2)
    label = int(self.are_connected(edges, u, v, N))

    # 3. DSU states (O(M·N), ~150 мкс для N=20, M=30)
    comp_states = self.compute_dsu_states(shuffled, N)

    # 4. Собрать input_ids + padding
    input_ids = self.encode(shuffled, u, v, label)

    return {
        'input_ids': input_ids,              # [seq_len]
        'target': label,                      # 0 или 1
        'answer_pos': answer_pos,             # int
        'edge_positions': edge_positions,     # [M]
        'comp_states': pad(comp_states, max_N), # [M, max_N]
        'num_vertices': N,                    # для маскирования
    }
```

**Производительность:** DSU для одного примера ~150 мкс. DataLoader с `num_workers=4` не будет бутылочным горлышком — forward pass GPU занимает десятки мс на батч.

**Стратегия val/test:** В отличие от train set, **validation и test** примеры **фиксированы** на диске (конкретный shuffle рёбер + конкретный query + label для каждого графа сохранены заранее). Это обеспечивает воспроизводимость метрик между запусками.

### 4.3 Типы графов (train)

| Тип | Параметры | Доля в train set |
|---|---|---|
| Erdos-Renyi G(N, p) | p ∈ {0.05, 0.10, 0.15, 0.20, 0.30} | 50% |
| Random trees | N-1 рёбер, случайное дерево (Prufer sequence) | 20% |
| Sparse random | m = N * k рёбер, k ∈ {1.0, 1.5, 2.0} | 20% |
| Complete graph (малые N) | K_n, n ∈ {3,4,5} | 5% |
| Path graph | Цепочка 0-1-2-..-(N-1) | 5% |

### 4.4 Размеры графов

- **Train:** N ∈ {8, 10, 12, 15, 20}
- **Validation / Test (in-distribution):** Те же N, те же типы, другие random seeds

### 4.5 OOD Test sets

| OOD Test Set | Описание | Цель |
|---|---|---|
| **Large N** | N ∈ {25, 30} | Генерализация по размеру |
| **Cyclic Grids** | Решётчатые графы (grid), cycles | Топология, не встречавшаяся в train |
| **Adversarial Degree** | hub + isolated components | Сломать degree-based heuristic |
| **Long Diameter** | Длинные цепочки + шум (диаметр >> 4) | Сломать shallow message passing |
| **Disconnected Dense** | Два плотных компонента без связи | Сломать density-based heuristic |

### 4.6 Баланс классов

~50/50 по ответу. Для каждого графа случайно выбираем query (u,v) с балансировкой.

### 4.7 Размеры датасетов

| Split | Размер (графов) |
|---|---|
| Train | 20,000 графов (∞ примеров через augmentation) |
| Validation | 2,000 графов |
| Test (in-distribution) | 2,000 графов |
| Test (OOD, каждый) | 1,000 графов |

---

## 5. Forward Pass (один батч)

```
  input_ids [B, seq_len]      <START> E01 E23 E12 E34 <SEP> Q03 <ANS> 1
                                        │    │    │    │                │
                                  GPT-2 (4L4H) — один forward pass
                                        │
                        ┌───────────────┴───────────────────┐
                        ▼                                   ▼
              attn_heads последнего слоя          hidden state на позиции <ANS>
              на позициях рёбер [t=1,2,3,4]              │
                        │                          Linear → 2 класса
                        │                                │
                  Probe (linear)                     L_LM (CE)
              W_h @ attn_h(t) → [max_N, max_N]
                        │
          CE vs true comp[] (masked по N)
                        │
                    L_state

                L = L_LM + λ * L_state — один backward pass
```

**В коде (батчевые операции, без циклов):**

```python
# 1. Forward — один вызов модели
hidden, attn_heads = model(input_ids)
# attn_heads: [B, H, seq_len, d_head]

# 2. L_LM
answer_logits = classifier(hidden[torch.arange(B), answer_pos])  # [B, 2]
L_lm = F.cross_entropy(answer_logits, target)

# 3. L_state — gather + matmul, без циклов
edge_attn = attn_heads[:, :, edge_positions, :]   # [B, H, M, d_head]
pred = probe(edge_attn)                            # [B, H, M, max_N, max_N]
L_state = masked_cross_entropy(pred, comp_states, vertex_mask)

# 4. Total
loss = L_lm + lambda_ * L_state
loss.backward()
```

---

## 6. Стадии экспериментов

### Стадия 1: Baseline SFT

**Цель:** Показать, что стандартный SFT застревает в локальном оптимуме / выучивает эвристики.

**Настройка:**
- Loss: только `L_LM` (cross-entropy на ответ)
- Без auxiliary loss
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

**Цель:** Главный эксперимент — показать, что DSU auxiliary loss значительно улучшает accuracy.

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

## 7. Метрики

| Метрика | Описание |
|---|---|
| **Accuracy (exact match)** | Доля правильных ответов 0/1 |
| **ID Accuracy** | Accuracy на in-distribution test set |
| **OOD Accuracy** | Accuracy на каждом OOD test set отдельно |
| **State-tracking Accuracy** | Доля правильно предсказанных component IDs probe-ом |
| **Degree Correlation** | Корреляция Спирмена между degree вершины и P(reachable) |
| **Per-diameter Accuracy** | Accuracy в зависимости от расстояния между query вершинами |
| **Loss curves** | L_LM, L_state, L_total по шагам |

---

## 8. Гиперпараметры (сводка)

| Параметр | Значение |
|---|---|
| Архитектура | 4-layer, 4-head GPT-2, d_model=256 |
| max_N (padding) | 30 |
| Optimizer | AdamW |
| LR | 5e-5 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Max sequence length | ~60 tokens (для N=20, m≈30 edges + query + specials) |
| Train graphs | 20,000 (∞ примеров через on-the-fly augmentation) |
| Val/Test graphs | 2,000 / 2,000 |
| lambda (aux loss weight) | Grid search: {0.1, 0.5, 1.0, 2.0, 5.0} |
| Early stopping | По val L_total, patience = 5-10 эпох |
| Training epochs | До 50 (с early stopping) |
| GPU | 1x (A100/H100/что доступно) |

---

## 9. Ожидаемые результаты (сводная таблица)

| Модель | ID Acc | OOD Large N | OOD Adversarial | OOD Long Diam |
|---|---|---|---|---|
| SFT (4L4H) | 70-90% | 50-60% | ~50% | ~50% |
| **Aux Loss (4L4H)** | **99-100%** | **85-99%** | **80-95%** | **75-90%** |
| Random Aux (control) | 70-90% | 50-60% | ~50% | ~50% |

---

## 10. План реализации (порядок)

1. **Генерация данных** — скрипт `generate_data.py`
   - Генерация графов всех типов (edge lists + N)
   - Сохранение как JSON / pickle

2. **DataLoader** — on-the-fly augmentation
   - Shuffle рёбер + random query + DSU states на лету
   - Padding / masking для переменного N

3. **Модель + SFT training** — базовый пайплайн
   - GPT-2 from scratch с compound tokenization
   - Training loop с логированием

4. **Auxiliary loss** — добавить probe head
   - Linear probe к attention outputs последнего слоя
   - Masked cross-entropy для comp[] states
   - Подбор lambda

5. **Evaluation pipeline** — автоматический запуск всех тестов
   - ID/OOD accuracy
   - State-tracking metrics

6. **Analysis & Interpretability** — logit attribution, probing, attention patterns
   - Jupyter notebooks для визуализации
   - Сравнение механизмов SFT vs Aux Loss

---

## 11. Риски и fallback-планы

| Риск | Вероятность | Mitigation |
|---|---|---|
| 4L4H не хватает для графов N=20 | Средняя | Увеличить до 6L4H; уменьшить N до 10-15 |
| DSU state слишком сложен для linear probe | Средняя | Использовать MLP probe (2 слоя) вместо linear |
| SFT работает слишком хорошо (нет проблемы) | Низкая | Увеличить N или усложнить OOD tests |
| Compute ограничения | Средняя | Уменьшить N, уменьшить train set, использовать fp16 |
| Aux Loss не помогает | Низкая | Проверить: правильный ли таргет? MLP probe? Другой слой? |
