# 💳 ML Pet Project: Кредитный скоринг (Home Credit Default Risk)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow.svg)](https://catboost.ai/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791.svg)](https://www.postgresql.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM%20%2F%20Core-red.svg)](https://www.sqlalchemy.org/)

Пет-проект по машинному обучению, посвященный разработке системы **кредитного скоринга** и прогнозированию вероятности невозврата займа (дефолта) клиентом банка. Проект построен на данных соревнования [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk).

---

## 📌 Содержание
- [Цель и постановка задачи](#-цель-и-постановка-задачи)
- [Данные](#-данные)
- [Сравнение моделей и метрики](#-сравнение-моделей-и-метрики)
- [Архитектура и пайплайн предобработки](#-архитектура-и-пайплайн-предобработки)
- [Структура репозитория](#-структура-репозитория)
- [Стек технологий](#-стек-технологий)
- [Быстрый старт](#-быстрый-старт)
- [Планы по развитию (Roadmap)](#-планы-по-развитию-roadmap)

---

## 🎯 Цель и постановка задачи

**Бизнес-цель:** Минимизировать финансовые потери банка за счет своевременного выявления потенциальных неплательщиков, сохраняя при этом высокий уровень одобрения надежным заемщикам.

**ML-постановка задачи:**
- **Тип задачи:** Бинарная классификация в условиях сильного дисбаланса классов (~8.07% дефолтов в обучающей выборке).
- **Целевая переменная (`TARGET`):**
  - `1` — клиент допустил просрочку платежа более чем на $X$ дней по первым $Y$ платежам (дефолт / трудности с выплатами).
  - `0` — все остальные случаи (успешное обслуживание кредита).
- **Ключевые метрики:**
  - **ROC-AUC** — основная метрика ранжирования вероятностей дефолта.
  - **PR-AUC (Average Precision)** — критически важная метрика качества ранжирования редкого положительного класса при сильном дисбалансе.
  - **F1-Score / Log-loss** — мониторинг точности калибровки и баланса Precision/Recall.

---

## 📊 Данные

Данные хранятся в реляционной базе данных **PostgreSQL** (`ml_pet_project`) и выгружаются через **SQLAlchemy**.

Основная таблица моделирования — `application_train` (~307,511 строк, 122 признака):
- Социально-демографические характеристики: возраст, пол, образование, семья, жилищные условия.
- Финансовые показатели: сумма кредита, доход, аннуитет, стоимость товара.
- Внешние скоринговые оценки: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`.
- Данные о занятости, документы, запросы в Бюро кредитных историй (БКИ).

> ℹ️ Подробное описание всех 122 колонок представлено в файле [descriptions.md](descriptions.md).

---

## 📈 Сравнение моделей и метрики

Все модели оценивались на валидационной выборке (тестовый сплит **25%** со стратификацией по целевой переменной, `random_state = 777`).

| Этап | Ноутбук | Модель | ROC-AUC | PR-AUC | Особенности решения |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **0. Baseline** | [`1_dummy_baseline.ipynb`](1_dummy_baseline.ipynb) | Dummy (Most Frequent) | `0.5000` | `~0.0807` | Фиксация нижнего порога метрик (Accuracy: 0.9193, F1: 0.0000, Log Loss: 0.2805). |
| **1. Linear** | [`2_linear_baseline.ipynb`](2_linear_baseline.ipynb) | Logistic Regression | `0.7482` | `0.2149` | `class_weight='balanced'`, кастомный `EmploymentCleaner`, логарифмирование дохода, OHE редких категорий, медианный импьютинг, `StandardScaler`. |
| **2. Boosting** | [`3_tree_baseline.ipynb`](3_tree_baseline.ipynb) | **CatBoost Classifier** | **`0.7601`** | **`0.2364`** | `auto_class_weights='Balanced'`, нативная обработка `cat_features`, ранняя остановка (`early_stopping_rounds=50`), оптимизация `eval_metric='AUC'`. |

### Динамика улучшений:
- Переход от наивного прогноза к **Logistic Regression** дал сильный прирост качества (ROC-AUC с `0.5000` до `0.7482`).
- Градиентный бустинг **CatBoost** превзошел линейную модель, подняв **ROC-AUC до 0.7601** и **PR-AUC до 0.2364** (лучшая итерация: 184).

---

## 🛠 Архитектура и пайплайн предобработки

В проекте реализован воспроизводимый конвейер подготовки признаков с использованием `scikit-learn.pipeline.Pipeline` и `ColumnTransformer`:

1. **Обработка аномалий занятости (`EmploymentCleaner`):**
   - Обнаружена аномалия в `DAYS_EMPLOYED = 365243` (порядка 1000 лет стажа — специфика кодирования отсутствия занятости/пенсионеров).
   - Аномальное значение заменяется на `NaN`, и создается бинарный флаг занятости `FLAG_WORKING = (DAYS_EMPLOYED != 365243)`.
2. **Трансформация финансовых признаков (`income_pipeline`):**
   - Логарифмирование `AMT_INCOME_TOTAL` через `FunctionTransformer(np.log1p)` для снижения влияния тяжелого правого хвоста распределения.
   - Заполнение пропусков медианой + масштабирование `StandardScaler`.
3. **Числовые признаки (`num_pipeline`):**
   - Медианный импьютинг `SimpleImputer(strategy='median')`.
   - Масштабирование признаков `StandardScaler()`.
4. **Категориальные признаки (`cat_pipeline`):**
   - Заполнение отсутствующих значений строкой `'Missing'`.
   - Кодирование категорий: `OneHotEncoder(drop='first', min_frequency=0.01, handle_unknown='ignore')` (для линейной модели) и нативный `cat_features` для CatBoost.

---

## 📂 Структура репозитория

```text
├── 1_dummy_baseline.ipynb     # Фиксация нижних порогов качества и наивного классификатора
├── 2_linear_baseline.ipynb    # Полноценный пайплайн с LogisticRegression и ColumnTransformer
├── 3_tree_baseline.ipynb      # Обучение градиентного бустинга CatBoostClassifier
├── config.py                  # Глобальные параметры конфигурации (random_state)
├── descriptions.md            # Справочник-описание всех полей исходного датасета
├── .env                       # Переменные окружения (пароли к БД) — не коммитится
├── .gitignore                 # Исключение артефактов, данных и виртуального окружения
├── data/                      # Исходные CSV-файлы Kaggle (хранятся локально)
└── sandbox/                   # Исследовательские скрипты и эксперименты со схемами БД
```

---

## 💻 Стек технологий

- **Язык программирования:** Python 3.10+
- **Машинное обучение:** Scikit-Learn, CatBoost
- **Анализ данных и вычисления:** Pandas, NumPy
- **Базы данных и ORM:** PostgreSQL, SQLAlchemy, psycopg2, SQLite
- **Управление окружением и конфигурацией:** python-dotenv, config.py

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/iqbara/ml_pet_project.git
cd ml_pet_project
```

### 2. Создание и активация виртуального окружения
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей
```bash
pip install pandas numpy scikit-learn catboost sqlalchemy psycopg2-binary python-dotenv jupyter
```

### 4. Настройка подключения к базе данных
Создайте файл `.env` в корневой директории проекта:
```env
DB_PASSWORD=your_postgresql_password
```
База данных по умолчанию: `ml_pet_project` (пользователь `postgres`, хост `localhost:5432`).

### 5. Запуск Jupyter Notebook
```bash
jupyter notebook
```
Последовательность запуска экспериментов:
1. `1_dummy_baseline.ipynb`
2. `2_linear_baseline.ipynb`
3. `3_tree_baseline.ipynb`

---

## 🗺 Планы по развитию (Roadmap)

- [ ] **Feature Engineering из реляционных таблиц:**
  - Агрегация кредитной истории из `bureau.csv` и `bureau_balance.csv` (активные кредиты, длительность просрочек).
  - Поведенческие признаки из истории предыдущих заявок `previous_application.csv`.
  - Платежная дисциплина по картам и рассрочкам (`installments_payments.csv`, `credit_card_balance.csv`).
- [ ] **Эксперименты с другими алгоритмами:**
  - Обучение LightGBM и XGBoost.
  - Подбор гиперпараметров с помощью Optuna.
  - Построение ансамбля / блендинга / стекинга лучших моделей.
- [ ] **Анализ интерпретируемости:**
  - SHAP (SHapley Additive exPlanations) анализ важности признаков.
  - Построение матриц ошибок и подбор оптимального порога классификации под бизнес-метрики (Profit / Loss matrix).
- [ ] **MLOps и развертывание:**
  - Обертка модели в сервис скоринга в реальном времени с помощью **FastAPI**.
  - Контейнеризация проекта в **Docker**.
  - Трекинг экспериментов через **MLflow**.
