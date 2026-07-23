#!/usr/bin/env python
# coding: utf-8

# ## Договорённости и протокол (актуальная версия)
# 
# ### Цель задачи
# 
# * Обучить и сравнить модели, которые разделяют **NBRFI** (позитив) и **None** (негатив) на уровне **каналов**.
# * Класс **NoneWNBRFI** **не используем в обучении**: он нужен как **hard negatives / стресс-тест**, чтобы проверить, не “стреляет” ли модель по каналам внутри `bright_NBRFI`, но вне частотных диапазонов RFI.
# 
# ---
# 
# ### Что такое классы в данных
# 
# * **NBRFI** — каналы, попадающие в частотные диапазоны RFI внутри сегментов, помеченных `bright_NBRFI`.
# * **None** — каналы из сегментов, где исходная метка сегмента была отсутствующей (`NaN`), и мы трактуем это как “нет RFI”.
# 
#   * В CSV это отражено в `label`, где `NaN` мы **явно заменяем на строку `"None"`**.
# * **NoneWNBRFI** — каналы внутри `bright_NBRFI` сегментов, но вне частотных диапазонов RFI: сложные отрицательные, похожие на позитивы по контексту.
# 
# ---
# 
# ### Признаки (что реально используем сейчас)
# 
# Используем фиксированный набор числовых статистик (16 колонок):
# 
# * `mean_o, std_o, skew_o, kurt_o`
# * `mean_n, std_n, skew_n, kurt_n`
# * `mean_o_ratio, std_o_ratio, skew_o_ratio, kurt_o_ratio`
# * `mean_n_ratio, std_n_ratio, skew_n_ratio, kurt_n_ratio`
# 
# Таргет:
# 
# * `y = 1`, если `label == "NBRFI"`
# * `y = 0`, если `label == "None"`
# 
# **Важно про NaN:** мы исправляем NaN **только в `label`**, чтобы не было скрытой деградации типов в признаках (числа не должны превращаться в `object`).
# 
# ---
# 
# ### Почему отсутствие кластеров на 2D-scatter не страшно
# 
# * Разделимость может проявляться:
# 
#   * только в **многомерной комбинации** признаков,
#   * или быть **нелинейной** (не видна в проекциях).
# * Поэтому 2D-графики — это диагностика, но не финальный критерий.
# 
# ---
# 
# ### Критически важный момент: anti-leakage split по группам
# 
# * “Объект” в датасете — **канал**, но каналы внутри одного сегмента сильно зависимы.
# * Поэтому делаем **групповой split** (по `segment_index`/`global_index` на этапе подготовки сплитов), и используем готовые индексы:
# 
#   * `train_idx`, `val_idx`, `test_idx`
# * Требование: все каналы одного сегмента попадают целиком **только в один сплит**.
# * Это особенно важно из-за `*_ratio` (канал/сегмент), которые усиливают риск утечки при случайном строковом split.
# 
# ---
# 
# ### Стратегия работы с большим объёмом (первый этап)
# 
# * Полный массив большой, поэтому для быстрого цикла используем **сбалансированный поднабор**:
# 
#   * примерно `30k NBRFI + 30k None` (или аналогичный баланс).
# * `NoneWNBRFI` в обучение не включаем.
# 
# ---
# 
# ### Правильный порядок формирования данных
# 
# 1. Сначала строим **групповые** `train/val/test` индексы по сегментам.
# 2. Затем внутри train-сегментов сэмплируем нужное число каналов каждого класса (баланс).
# 3. Val/Test формируем аналогично (обычно меньше для скорости).
# 
# ---
# 
# ## Протокол обучения и сравнения моделей (то, что мы реализовали)
# 
# ### Общий pipeline
# 
# * `X` = 16 числовых фич.
# * Для большинства моделей применяем `StandardScaler` (деревьям он не нужен).
# * Для каждой модели:
# 
#   1. обучаемся на `train`
#   2. считаем вероятности/скор на `val`
#   3. **подбираем порог на `val`** (в текущей реализации — по максимуму F1)
#   4. оцениваем при этом пороге на `test`
#   5. складываем всё в сравнительную таблицу
# 
# ### Набор моделей (лестница сложности)
# 
# Мы сравниваем семейства, чтобы понять, какой тип границы нужен:
# 
# 1. **SGD_LogReg** — линейная логистическая регрессия (SGD, `log_loss`)
#    Проверяет: есть ли **линейный** сигнал в текущих фичах.
# 
# 2. **SGD_LinearSVM** — линейный SVM (SGD, `hinge`)
#    Проверяет: помогает ли “margin-критерий” при той же линейной границе.
# 
# 3. **Poly2_LogReg** — полиномиальные признаки степени 2 + линейная логрег
#    Проверяет: есть ли выигрыш от простых нелинейностей вида взаимодействий `x_i*x_j`.
# 
# 4. **RBFapprox_LogReg** — аппроксимация RBF-ядра (Random Fourier Features) + линейная логрег
#    Проверяет: нужен ли “RBF-тип” нелинейности (похоже на RBF-SVM, но масштабируемо).
# 
# 5. **HistGB** — градиентный бустинг по деревьям
#    Проверяет: насколько хорошо табличная нелинейная модель “вытаскивает” сигнал без ручных фич-маппингов.
# 
# 6. **MLP** — небольшая нейросеть
#    Проверяет: сможет ли универсальная нелинейная модель выучить нужные комбинации признаков.
# 
# ---
# 
# ### Метрики и порог
# 
# * На первом этапе (baseline/сравнение) используем метрики:
# 
#   * ROC-AUC, PR-AUC (качество ранжирования)
#   * logloss (качество вероятностей/уверенности)
#   * precision/recall/F1/accuracy **при выбранном пороге**
# * Порог выбираем на `val` (сейчас: **max F1** как простая автоматизация).
# * Дальше (когда подключим NoneWNBRFI) порог будем подбирать уже под ограничения на FP.
# 
# ---
# 
# ## Протокол оценки качества — два режима
# 
# 1. **Основная валидация:** качество на задаче `NBRFI vs None` (train/val/test).
# 2. **Стресс-тест:** прогоняем модель на **NoneWNBRFI** и считаем:
# 
#    * FP-rate при фиксированном пороге,
#    * распределение `p(NBRFI)` для NoneWNBRFI vs None vs NBRFI,
#    * “насколько путает hard negatives с позитивом”.
# 
# ---
# 
# ## Что делаем дальше
# 
# * Запускаем сравнение моделей на текущем сбалансированном поднаборе и фиксируем таблицу результатов.
# * По таблице выбираем 2–3 самых перспективных семейства (например: лучший линейный + лучший нелинейный).
# * Добавляем **стресс-тест на NoneWNBRFI**:
# 
#   * графики распределений скоринга,
#   * FP-rate при выбранном пороге,
#   * подбор порога уже под ограничение FP на hard negatives.
# * После первых результатов:
# 
#   * расширяем поднабор / усложняем фичи / тюним гиперпараметры.
# 
# ---
# 
# Если хочешь, я ещё могу “свести” это в короткую “шапку ноутбука” (Markdown-ячейку) и отдельно сделать “чек-лист запуска” (что должно лежать рядом: csv, npz, список фич).
# 

# In[69]:


import glob
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, log_loss,
    precision_recall_fscore_support
)

from tqdm.notebook import tqdm, trange


# In[5]:


meta_path  = "B0531+21_59000_48386_subset_channels_meta.csv"
split_path = "split_indices.npz"

meta = pd.read_csv(meta_path)

# ВАЖНО:
# Ты говорил, что NaN только в label и это "None". Исправляем ТОЛЬКО label,
# чтобы случайно не превратить числовые фичи в object.
meta["label"] = meta["label"].fillna("None")

splits = np.load(split_path)
train_idx = splits["train_idx"]
val_idx   = splits["val_idx"]
test_idx  = splits["test_idx"]

# Явно задаём список фичей (как ты прислал)
feature_cols = [
    'mean_o', 'std_o', 'skew_o', 'kurt_o',
    'mean_n', 'std_n', 'skew_n', 'kurt_n',
    'mean_o_ratio', 'std_o_ratio', 'skew_o_ratio', 'kurt_o_ratio',
    'mean_n_ratio', 'std_n_ratio', 'skew_n_ratio', 'kurt_n_ratio'
]

# На всякий случай проверим, что все колонки есть
missing_cols = [c for c in feature_cols if c not in meta.columns]
if missing_cols:
    raise ValueError(f"Нет колонок в meta: {missing_cols}")

# y: NBRFI=1, None=0
y_all = (meta["label"].values == "NBRFI").astype(int)

# X: только нужные признаки
X_all = meta[feature_cols].copy()

# Приведём к числам (на случай если где-то в csv числа лежат строками)
for c in feature_cols:
    X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

print("meta shape:", meta.shape, "(rows=объекты/примеры)")
print("X_all shape:", X_all.shape)
print("labels counts:", np.unique(y_all, return_counts=True))
print("train/val/test:", len(train_idx), len(val_idx), len(test_idx))

# Проверка: нет ли NaN в признаках (ты говорил, что их нет)
nan_total = int(X_all.isna().sum().sum())
print("Total NaN in X_all:", nan_total)
if nan_total > 0:
    # не ломаем пайплайн, но явно предупреждаем
    print("WARNING: обнаружены NaN в признаках. Это не ожидалось по твоим данным.")

X_train = X_all.iloc[train_idx].reset_index(drop=True)
X_val   = X_all.iloc[val_idx].reset_index(drop=True)
X_test  = X_all.iloc[test_idx].reset_index(drop=True)

y_train = y_all[train_idx]
y_val   = y_all[val_idx]
y_test  = y_all[test_idx]


# ### утилиты: выбор порога по val и расчёт метрик

# In[6]:


def best_threshold_by_f1(y_true, p, grid=None):
    """
    Подбор порога по валидации.

    Что делаем:
    - Берём вероятности p (модельный скор 0..1).
    - Пробуем разные пороги.
    - Выбираем порог, который даёт максимум F1 на val.

    Что проверяем:
    - Можно ли "выжать" лучшее качество простым подбором порога,
      не меняя модель.
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def eval_binary(y_true, p, threshold=0.5):
    """
    Считаем набор метрик для бинарной классификации.

    Что делаем:
    - p: вероятности класса 1 (NBRFI).
    - threshold: порог, выше которого считаем NBRFI.

    Что проверяем:
    - качество ранжирования (ROC-AUC, PR-AUC)
    - калибровку/уверенность (logloss)
    - качество решения при заданном пороге (precision/recall/f1/accuracy)
    """
    pred = (p >= threshold).astype(int)

    acc = accuracy_score(y_true, pred)
    roc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan
    ap  = average_precision_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan

    # log_loss требует вероятности и наличие обоих классов
    ll = log_loss(y_true, np.c_[1-p, p], labels=[0, 1]) if len(np.unique(y_true)) == 2 else np.nan

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)

    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(ap),
        "logloss": float(ll),
    }


# ### набор моделей

# In[7]:


# Здесь важная идея:
# Мы строим "лестницу" моделей от простого к более сложному.
# Каждая следующая модель проверяет, есть ли выгода от:
# - линейной границы,
# - другой линейной потери (SVM-hinge),
# - явных взаимодействий признаков (полином),
# - нелинейности типа RBF (через аппроксимацию ядра),
# - табличного сильного baseline (градиентный бустинг),
# - небольшой нейросети (MLP) как универсальной нелинейной модели.

models = {}

# 1) Линейная логистическая регрессия, обученная SGD (как у тебя)
# Что делаем:
# - строим линейную границу w·x + b
# - переводим в вероятность через сигмоиду
# Что проверяем:
# - есть ли линейный сигнал в статистических фичах (mean/std/skew/kurt и ratio)
models["SGD_LogReg"] = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(
        loss="log_loss",      # логистический лосс = лог. регрессия
        alpha=1e-4,           # L2-регуляризация (чем больше, тем сильнее "сжимает" веса)
        max_iter=2000,
        tol=1e-3,
        random_state=42
    ))
])

# 2) Линейный SVM через SGD (hinge loss)
# Что делаем:
# - тоже линейная граница
# - но оптимизируем hinge loss (SVM-идея: максимизируем margin)
# Что проверяем:
# - поможет ли другой критерий (margin) вместо логлосса при тех же фичах
models["SGD_LinearSVM"] = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(
        loss="hinge",         # hinge = линейный SVM
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        random_state=42
    ))
])

# 3) Полиномиальные признаки (2 степень) + линейная логрег
# Что делаем:
# - расширяем пространство признаков: добавляем x_i*x_j и x_i^2
# - после этого линейная модель в новом пространстве == нелинейная граница в исходном
# Что проверяем:
# - есть ли простая нелинейность вида "взаимодействий" между статистиками
models["Poly2_LogReg"] = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("clf", SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        random_state=42
    ))
])

# 4) Аппроксимация RBF-ядра (Random Fourier Features) + линейная логрег
# Что делаем:
# - RBFSampler превращает x -> z(x) так, что линейная модель по z(x)
#   приближает RBF-SVM (нелинейную границу)
# Что проверяем:
# - нужен ли "RBF-тип" нелинейности для отделения NBRFI от None
models["RBFapprox_LogReg"] = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("rbf", RBFSampler(gamma=1.0, n_components=800, random_state=42)),
    ("clf", SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        random_state=42
    ))
])

# 5) Градиентный бустинг по деревьям (Histogram-based)
# Что делаем:
# - сильная табличная модель: строит ансамбль деревьев, умеет сложные границы,
#   автоматически ловит взаимодействия и нелинейности
# Что проверяем:
# - "если в данных есть сигнал", то бустинг часто вытягивает его лучше линейных моделей
models["HistGB"] = Pipeline(steps=[
    # скейлер не нужен деревьям, но не мешает; здесь уберём ради ясности:
    ("clf", HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42
    ))
])

# 6) MLP (небольшая нейросеть)
# Что делаем:
# - универсальная нелинейная модель, сама учит комбинации признаков
# Что проверяем:
# - есть ли нелинейная структура, которую проще поймать MLP, чем явным poly/RBF
models["MLP"] = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=42
    ))
])

print("Models:", list(models.keys()))


# ### обучение, подбор порога по val, оценка на test, сравнительная таблица

# In[10]:


results = []
fitted_models = {}

for name, pipe in tqdm(models.items(), total=len(models), desc="Training models"):
    t0 = time.time()

    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0

    if hasattr(pipe, "predict_proba"):
        p_val  = pipe.predict_proba(X_val)[:, 1]
        p_test = pipe.predict_proba(X_test)[:, 1]
    else:
        scores_val  = pipe.decision_function(X_val)
        scores_test = pipe.decision_function(X_test)

        p_val  = 1.0 / (1.0 + np.exp(-scores_val))
        p_test = 1.0 / (1.0 + np.exp(-scores_test))

    best_t, best_val_f1 = best_threshold_by_f1(y_val, p_val)

    val_metrics  = eval_binary(y_val,  p_val,  threshold=best_t)
    test_metrics = eval_binary(y_test, p_test, threshold=best_t)

    row = {
        "model": name,
        "fit_time_s": round(fit_time, 3),
        "best_thr_val": round(best_t, 3),

        "val_f1": round(val_metrics["f1"], 4),
        "val_recall": round(val_metrics["recall"], 4),
        "val_precision": round(val_metrics["precision"], 4),
        "val_roc_auc": round(val_metrics["roc_auc"], 4),
        "val_pr_auc": round(val_metrics["pr_auc"], 4),
        "val_logloss": round(val_metrics["logloss"], 4),

        "test_f1": round(test_metrics["f1"], 4),
        "test_recall": round(test_metrics["recall"], 4),
        "test_precision": round(test_metrics["precision"], 4),
        "test_roc_auc": round(test_metrics["roc_auc"], 4),
        "test_pr_auc": round(test_metrics["pr_auc"], 4),
        "test_logloss": round(test_metrics["logloss"], 4),
        "test_acc": round(test_metrics["acc"], 4),
    }

    results.append(row)
    fitted_models[name] = pipe

df = pd.DataFrame(results)
df_sorted = df.sort_values(["val_f1", "val_pr_auc"], ascending=False).reset_index(drop=True)
df_sorted


# In[11]:


key_cols = [
    "model", "fit_time_s", "best_thr_val",
    "val_f1", "val_precision", "val_recall", "val_roc_auc", "val_pr_auc",
    "test_f1", "test_precision", "test_recall", "test_roc_auc", "test_pr_auc", "test_acc"
]

df_sorted[key_cols]


# ### Поиск наиболее важных признаков

# In[13]:


def permutation_importance_all_models(fitted_models, X_val, y_val,
                                      scoring="average_precision",
                                      n_repeats=10, random_state=42):
    """
    Считаем permutation importance для каждой модели на val.

    Что делаем:
    - Для каждого признака перемешиваем его значения (ломаем информацию в этом столбце).
    - Меряем, насколько падает метрика scoring.
    - Чем больше падение => тем важнее признак.

    Почему scoring=average_precision:
    - это PR-AUC, порог-независимо, хорошо для ранжирующих моделей.
    """
    features = list(X_val.columns)
    imp_mean = pd.DataFrame(index=fitted_models.keys(), columns=features, dtype=float)
    imp_std  = pd.DataFrame(index=fitted_models.keys(), columns=features, dtype=float)

    for name, model in tqdm(fitted_models.items(), total=len(fitted_models), desc="Permutation importance"):
        perm = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring
        )
        imp_mean.loc[name, :] = perm.importances_mean
        imp_std.loc[name, :]  = perm.importances_std

    return imp_mean, imp_std

imp_mean, imp_std = permutation_importance_all_models(
    fitted_models, X_val, y_val,
    scoring="average_precision",
    n_repeats=10,
    random_state=42
)

imp_mean


# In[14]:


def topk_features_per_model(imp_mean, k=3):
    rows = []
    for model_name in imp_mean.index:
        s = imp_mean.loc[model_name].sort_values(ascending=False)
        top_feats = s.index[:k].tolist()
        top_vals  = s.values[:k].tolist()

        row = {"model": model_name}
        for i in range(k):
            row[f"top{i+1}_feat"] = top_feats[i]
            row[f"top{i+1}_imp"]  = float(top_vals[i])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("top1_imp", ascending=False).reset_index(drop=True)

top3_table = topk_features_per_model(imp_mean, k=3)
top3_table


# In[16]:


def fit_eval_subset_for_all(models, X_train, y_train, X_val, y_val, X_test, y_test, imp_mean, k=3):
    rows = []

    for name, base_model in tqdm(models.items(), total=len(models), desc=f"Refit on top-{k}"):
        # топ-k признаков именно для этой модели
        cols = imp_mean.loc[name].sort_values(ascending=False).index[:k].tolist()

        # --- full features ---
        m_full = clone(base_model)
        m_full.fit(X_train, y_train)

        if hasattr(m_full, "predict_proba"):
            p_val_full  = m_full.predict_proba(X_val)[:, 1]
            p_test_full = m_full.predict_proba(X_test)[:, 1]
        else:
            s_val_full  = m_full.decision_function(X_val)
            s_test_full = m_full.decision_function(X_test)
            p_val_full  = 1/(1+np.exp(-s_val_full))
            p_test_full = 1/(1+np.exp(-s_test_full))

        t_full, _ = best_threshold_by_f1(y_val, p_val_full)
        test_full = eval_binary(y_test, p_test_full, threshold=t_full)

        # --- top-k features ---
        m_sub = clone(base_model)
        m_sub.fit(X_train[cols], y_train)

        if hasattr(m_sub, "predict_proba"):
            p_val_sub  = m_sub.predict_proba(X_val[cols])[:, 1]
            p_test_sub = m_sub.predict_proba(X_test[cols])[:, 1]
        else:
            s_val_sub  = m_sub.decision_function(X_val[cols])
            s_test_sub = m_sub.decision_function(X_test[cols])
            p_val_sub  = 1/(1+np.exp(-s_val_sub))
            p_test_sub = 1/(1+np.exp(-s_test_sub))

        t_sub, _ = best_threshold_by_f1(y_val, p_val_sub)
        test_sub = eval_binary(y_test, p_test_sub, threshold=t_sub)

        rows.append({
            "model": name,
            "top_features": cols,
            "test_f1_full": round(test_full["f1"], 4),
            "test_f1_topk": round(test_sub["f1"], 4),
            "test_pr_auc_full": round(test_full["pr_auc"], 4),
            "test_pr_auc_topk": round(test_sub["pr_auc"], 4),
            "test_acc_full": round(test_full["acc"], 4),
            "test_acc_topk": round(test_sub["acc"], 4),
        })

    return pd.DataFrame(rows).sort_values("test_f1_topk", ascending=False).reset_index(drop=True)

subset_compare = fit_eval_subset_for_all(
    models, X_train, y_train, X_val, y_val, X_test, y_test,
    imp_mean, k=3
)

subset_compare


# In[43]:


K = 3

topk_by_model = {
    name: imp_mean.loc[name].sort_values(ascending=False).index[:K].tolist()
    for name in imp_mean.index
}

topk_by_model


# ### переобучить top-k модели и сохранить (joblib)

# In[44]:


Path("saved_models_stat").mkdir(exist_ok=True)

val_preds = {}  # сюда положим p_val и cols для каждой модели

for name, base_model in tqdm(models.items(), total=len(models), desc=f"Refit+Save top-{K}"):
    cols = topk_by_model[name]

    pipe_small = clone(base_model)
    pipe_small.fit(X_train[cols], y_train)

    if hasattr(pipe_small, "predict_proba"):
        p_val = pipe_small.predict_proba(X_val[cols])[:, 1]
    else:
        s_val = pipe_small.decision_function(X_val[cols])
        p_val = 1.0 / (1.0 + np.exp(-s_val))

    best_t, best_val_f1 = best_threshold_by_f1(y_val, p_val)

    bundle_small = {
        "model_name": f"{name}_top{K}",
        "base_model": name,
        "threshold": float(best_t),
        "feature_cols": cols,
        "pipeline": pipe_small,
        "val_f1_at_thr": float(best_val_f1),
    }

    out_path = f"saved_models_stat/rfi_model_{name}_top{K}.joblib"
    joblib.dump(bundle_small, out_path, compress=("xz", 3))

    val_preds[name] = {"p_val": p_val, "cols": cols, "thr": best_t}

print("Saved:", len(list(Path("saved_models_stat").glob("*.joblib"))), "files")


# In[45]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_val_curves(y_true, p, title="", thr=None):
    # ROC
    fpr, tpr, roc_thr = roc_curve(y_true, p)
    roc_auc = auc(fpr, tpr)

    # PR
    prec, rec, pr_thr = precision_recall_curve(y_true, p)
    pr_auc = auc(rec, prec)

    # F1 vs threshold (строим по сетке порогов)
    thr_grid = np.linspace(0.01, 0.99, 99)
    f1s = []
    for t in thr_grid:
        pred = (p >= t).astype(int)
        pr, re, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        f1s.append(f1)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title} | ROC AUC={roc_auc:.3f}")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} | PR AUC={pr_auc:.3f}")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(thr_grid, f1s)
    if thr is not None:
        plt.axvline(thr, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title(f"{title} | F1 vs threshold")
    plt.grid(True)
    plt.show()


# In[46]:


for name in val_preds.keys():
    plot_val_curves(y_val, val_preds[name]["p_val"], title=f"{name}_top{K}", thr=val_preds[name]["thr"])


# ### Тестирвоание моделей 

# In[47]:


models_list = glob.glob('./saved_models_stat/*.joblib')
models_list


# In[48]:


import glob, os

for f in sorted(glob.glob("saved_models_stat/*.joblib"))[:5]:
    size_mb = os.path.getsize(f)/1024/1024
    b = joblib.load(f)
    print(f, f"| {size_mb:.2f} MB | cols={b['feature_cols']} | thr={b['threshold']:.3f}")


# In[49]:


bundle = joblib.load(models_list[1]) 

model_name   = bundle["model_name"]
threshold    = float(bundle["threshold"])
feature_cols = bundle["feature_cols"]
pipe         = bundle["pipeline"]

print("Loaded model:", model_name)
print("Threshold:", threshold)
print("Features:", feature_cols)


# In[50]:


def compute_stats(x: np.ndarray, axis=None):
    """
    Возвращает mean, std, skew, kurt (excess) по заданной оси.
    """
    x = np.asarray(x)
    mean = x.mean(axis=axis, keepdims=True)
    std  = x.std(axis=axis, keepdims=True)

    centered = x - mean
    m2 = (centered ** 2).mean(axis=axis, keepdims=True)
    m3 = (centered ** 3).mean(axis=axis, keepdims=True)
    m4 = (centered ** 4).mean(axis=axis, keepdims=True)

    std_safe = np.where(std == 0, 1.0, std)
    skew = m3 / (std_safe ** 3)
    kurt = m4 / (std_safe ** 4) - 3.0

    return mean.squeeze(), std.squeeze(), skew.squeeze(), kurt.squeeze()


def normalize_per_channel(seg: np.ndarray) -> np.ndarray:
    """
    Нормализация каждого частотного канала (строки) в [0, 256].
    seg: (n_channels, n_time)
    """
    seg = np.asarray(seg)
    chan_min = seg.min(axis=1, keepdims=True)
    chan_max = seg.max(axis=1, keepdims=True)
    denom = chan_max - chan_min
    denom_safe = np.where(denom == 0, 1.0, denom)
    return (seg - chan_min) / denom_safe * 256.0


def compute_ratio(per_channel_vals: np.ndarray, seg_val: float) -> np.ndarray:
    """
    Отношение параметра канала к параметру всего сегмента.
    Если seg_val == 0, возвращаем NaN.
    """
    per_channel_vals = np.asarray(per_channel_vals, dtype=float)
    if seg_val == 0:
        return np.full_like(per_channel_vals, np.nan, dtype=float)
    return per_channel_vals / seg_val


def extract_features_from_segment(seg: np.ndarray) -> pd.DataFrame:
    """
    seg: (n_channels, 256)  - как у тебя data после get_data_from_filterbank(...).T
    Возвращает DataFrame (n_channels, 16) с колонками:
      mean_o, std_o, skew_o, kurt_o,
      mean_n, std_n, skew_n, kurt_n,
      mean_o_ratio, ... kurt_n_ratio
    """
    seg = np.asarray(seg)
    if seg.ndim != 2:
        raise ValueError(f"Expected seg with shape (n_channels, n_time), got {seg.shape}")

    # 1) per-channel stats on original (axis=1 => по времени)
    mean_o, std_o, skew_o, kurt_o = compute_stats(seg, axis=1)

    # 2) normalized per-channel segment + per-channel stats
    seg_n = normalize_per_channel(seg)
    mean_n, std_n, skew_n, kurt_n = compute_stats(seg_n, axis=1)

    # 3) segment-level stats (по всем значениям сегмента)
    seg_mean_o, seg_std_o, seg_skew_o, seg_kurt_o = compute_stats(seg, axis=None)
    seg_mean_n, seg_std_n, seg_skew_n, seg_kurt_n = compute_stats(seg_n, axis=None)

    # 4) ratios
    mean_o_ratio = compute_ratio(mean_o, seg_mean_o)
    std_o_ratio  = compute_ratio(std_o,  seg_std_o)
    skew_o_ratio = compute_ratio(skew_o, seg_skew_o)
    kurt_o_ratio = compute_ratio(kurt_o, seg_kurt_o)

    mean_n_ratio = compute_ratio(mean_n, seg_mean_n)
    std_n_ratio  = compute_ratio(std_n,  seg_std_n)
    skew_n_ratio = compute_ratio(skew_n, seg_skew_n)
    kurt_n_ratio = compute_ratio(kurt_n, seg_kurt_n)

    df = pd.DataFrame({
        "mean_o": mean_o, "std_o": std_o, "skew_o": skew_o, "kurt_o": kurt_o,
        "mean_n": mean_n, "std_n": std_n, "skew_n": skew_n, "kurt_n": kurt_n,
        "mean_o_ratio": mean_o_ratio, "std_o_ratio": std_o_ratio,
        "skew_o_ratio": skew_o_ratio, "kurt_o_ratio": kurt_o_ratio,
        "mean_n_ratio": mean_n_ratio, "std_n_ratio": std_n_ratio,
        "skew_n_ratio": skew_n_ratio, "kurt_n_ratio": kurt_n_ratio,
    })

    # защита: если вдруг возникли nan/inf — sklearn упадёт.
    # Если в твоих данных такого не бывает, этот блок просто ничего не меняет.
    if not np.isfinite(df.to_numpy()).all():
        bad = np.sum(~np.isfinite(df.to_numpy()))
        print(f"WARNING: found non-finite values in features: {bad}. Replacing with 0.0")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


# In[51]:


import your
from astropy import units as u

def get_data_from_filterbank(filterbank_file, index, rand=0):
    steps = range(0, filterbank_file.your_header.nspectra, 256)
    return filterbank_file.get_data(nstart=steps[index] + rand, nsamp=256).T  # (n_chan, 256)

filterbank_file_path = "/hercules/results/akazantsev/filterbank_files/B0531+21/B0531+21_59000_48386.fil"  # <- поменяй
filterbank_file = your.Your(filterbank_file_path)

f_first = filterbank_file.your_header.fch1 * u.megahertz
f_last  = (filterbank_file.your_header.fch1 + filterbank_file.your_header.bw) * u.megahertz
tsamp   = filterbank_file.your_header.tsamp * u.second

print("f_first:", f_first, "f_last:", f_last, "tsamp:", tsamp)


# In[64]:


def predict_mask_for_segment(seg: np.ndarray, pipe, threshold: float, feature_cols: list):
    """
    Возвращает:
      row_mask_bool: (n_channels,) bool
      probs: (n_channels,) float
      feats_df: (n_channels, 16) DataFrame
    """
    feats = extract_features_from_segment(seg)
    # гарантируем порядок колонок как при обучении
    feats = feats[feature_cols]

    # вероятности
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(feats)[:, 1]
    else:
        # на случай hinge-модели без predict_proba
        scores = pipe.decision_function(feats)
        probs = 1.0 / (1.0 + np.exp(-scores))

    mask = probs >= threshold
    return mask, probs, feats


# In[80]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_spec_with_mask_sklearn(filterbank_file, index, pipe, threshold, feature_cols, plot=False,
                                save=False, random_shift=False, random_seed=42):
    data = get_data_from_filterbank(filterbank_file, index)  # (n_chan, 256)
    dy = dx = 0

    rng = np.random.default_rng(random_seed)
    if random_shift:
        dy = int(rng.integers(50, 150))
        dx = int(rng.integers(50, 150))
        data = np.roll(data, shift=(dy, dx), axis=(0, 1))

    row_mask, probs, feats = predict_mask_for_segment(data, pipe, threshold, feature_cols)
    H, W = data.shape
    overlay = np.repeat(row_mask[:, None], W, axis=1).astype(float)

    cmap_black = ListedColormap([(0, 0, 0, 0.0), (0, 0, 0, 1.0)])
    vmin, vmax = float(np.min(data)), float(np.max(data))

    extent = [0, W, float(f_last.value), float(f_first.value)]

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(
            f"Index {index} | Model: {model_name} | thr={threshold:.3f}\nShift: dx={dx}, dy={dy}",
            fontsize=13
        )
    
        ax1.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, extent=extent, aspect="auto", origin="upper")
        ax1.set_title("Original")
        ax1.set_xlabel(f"Time samples (dt={tsamp.to(u.microsecond):.1f})")
        ax1.set_ylabel("Frequency")
    
        ax2.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, extent=extent, aspect="auto", origin="upper")
        ax2.imshow(overlay, cmap=cmap_black, vmin=0, vmax=1, extent=extent, aspect="auto", origin="upper")
        ax2.set_title("With mask")
        ax2.set_xlabel(f"Time samples (dt={tsamp.to(u.microsecond):.1f})")
    
        plt.tight_layout()
    
        if save:
            out = f"spec_with_mask_{model_name}_idx_{index}_dx_{dx}_dy_{dy}.png"
            plt.savefig(out, dpi=250)
            print("Saved:", out)
        else:
            plt.show()
    
    return row_mask, probs, feats


# In[97]:


row_mask, probs, feats = plot_spec_with_mask_sklearn(
    filterbank_file,
    index=17053,
    pipe=pipe,
    threshold=threshold,     # можно поставить 0.9 как в CNN, но лучше использовать сохранённый
    feature_cols=feature_cols,
    plot=True,
    save=False,
    random_shift=False
)

print("Predicted RFI channels:", int(row_mask.sum()), "out of", row_mask.size)
print("probs summary:", np.min(probs), np.mean(probs), np.max(probs))


# In[68]:


np.array([True, True]).all()


# In[86]:


segments_with_false_channels = []
for i in trange(50000):
    row_mask, probs, feats = plot_spec_with_mask_sklearn(
        filterbank_file,
        index=i,
        pipe=pipe,
        threshold=threshold,     # можно поставить 0.9 как в CNN, но лучше использовать сохранённый
        feature_cols=feature_cols,
        save=False,
        random_shift=False
    )

    if not row_mask.all():
        segments_with_false_channels.append(i)


# In[87]:


print(len(segments_with_false_channels))


# In[88]:


segments_with_false_channels[:50] 


# In[ ]:




