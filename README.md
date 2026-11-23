# Рекомендательная система: Kaggle «Dz 1: Recommender System» (2026)

Задача: по логам музыкального сервиса (прослушивания, лайки, дизлайки) выдать каждому пользователю из `test_users.csv` до 10 треков, которые он **будет слушать после границы train**. Повторные прослушивания засчитываются. 

Метрика - NDCG@10, усреднённая по пользователям.

Решение двухстадийное:

1. **Отбор кандидатов.** Из ~880 тыс. треков для каждого пользователя оставляем несколько сотен правдоподобных: не больше ~550 (до 300 из истории, по 100 от i2i и iALS, 50 популярных), на деле меньше из-за пересечений источников. Точное число печатает `validate` (строка `candidates: ... (N/user)`).
2. **Ранжирование.** LightGBM LambdaRank сортирует этих кандидатов, топ-10 идёт в сабмит.

## Установка и запуск

```powershell
uv sync
```

Скачать со страницы соревнования (вкладка Data) `train.parquet` и `test_users.csv` в папку `data/`.

```powershell
uv run python main.py validate
```
```powershell
uv run python main.py submit
```

- `validate` считает локальный NDCG@10, подбирает число деревьев и сохраняет обучающий срез в `output/train_fold.parquet`.
- `submit` обучает финальный ранкер и пишет `output/submission.csv`.

Параметры командной строки: `--rounds N` (максимум деревьев при validate), `--final-rounds N` (число деревьев при submit).
Пути переопределяются переменными окружения `RECSYS_DATA` и `RECSYS_OUT`.

При запуске `validate` и `submit` в консоль печатается вход ранкера: размер таблицы кандидатов и 5 случайных строк, признаки по строкам. Самые важные признаки `validate` печатает в конце (`top features`).

## Структура проекта

Модули сгруппированы по папкам этапов `src/s<N>_<этап>/`. Имя пакета Python не может начинаться с цифры, поэтому перед номером стоит `s`.

| Файл | Этап | Что делает |
|---|---|---|
| `main.py` | - | Точка входа: вызывает этапы по порядку для `validate` и `submit` |
| `src/config.py` | - | Общее: пути, `DAY`, `SEED`, типы событий (`LISTEN`, `LIKE`, … и `pl.Enum` `EVENT_TYPE`) |
| `src/s<N>_<этап>/config.py` | 0–3 | Гиперпараметры своего этапа; список источников `SOURCES` - в `s2_candidates/config.py` |
| `src/utils.py` | - | Логирование и общие выражения polars |
| `src/s0_data/data.py` | 0 | Загрузка, граница `T_val`, `time_split`, метки, разбиение пользователей |
| `src/s1_features/past.py` | 1 | `PastEvents` (события до `T`) и `BaseFeatures` (признаки, не зависящие от источников) |
| `src/s1_features/tables.py` | 1 | Признаки пары пользователь×трек, пользователя, трека, реакций |
| `src/s2_candidates/candidates.py` | 2 | `build_candidates`: точка входа этапов 1–2, считает признаки, собирает кандидатов из источников |
| `src/s2_candidates/sources/base.py` | 2 | Интерфейс источника `Source`, реестр `SOURCE_CLASSES`, `make_sources` |
| `src/s2_candidates/sources/hist.py` | 2 | `HistSource`: треки из истории пользователя |
| `src/s2_candidates/sources/pop.py` | 2 | `PopSource`: популярное за 7 дней |
| `src/s2_candidates/sources/i2i.py` | 2 | `I2ISource`: item2item по совместной встречаемости в сессиях |
| `src/s2_candidates/sources/als.py` | 2 | `AlsSource`: iALS через `implicit`, веса уверенности, кандидаты, признаки |
| `src/s3_ranker/ranker.py` | 3 | `train_ranker`, `recommend` (LightGBM LambdaRank) |
| `src/s4_evaluation/metrics.py` | 4 | NDCG@10, полнота кандидатов, отчёт по источникам |
| `src/s4_evaluation/submission.py` | 4 | Сборка `submission.csv` |
| `scripts/make_synthetic_data.py` | - | Маленький синтетический датасет для проверки пайплайна |
Правила оформления кода и проверка перед изменениями описаны в [AGENTS.md](AGENTS.md).

## Данные

| Файл | Содержимое |
|---|---|
| `train.parquet` | 43 млн событий до общей границы, 9 844 пользователя. Колонки: `uid`, `timestamp` (секунды, со сдвигом), `item_id`, `is_organic`, `played_ratio_pct`, `track_length_seconds`, `event_type` (`listen`, `like`, `unlike`, `dislike`, `undislike`) |
| `test_users.csv` | 9 213 пользователей для сабмита; 151 из них нет в train |
| ответы теста | Скрыты на Kaggle: треки, прослушанные после границы (последние 10% всех прослушиваний). Пользователи поделены на Public и Private примерно поровну |

## Что дальше

- Подобрать веса уверенности iALS и `ALS_FACTORS` по полноте источника `als` и итоговому NDCG.
- Добавить второй, более ранний временной срез для обучения ранкера: вдвое больше данных и проверка стабильности во времени.
- Сдача: выбрать 2 сабмита для Private по **локальной** валидации, а не по Public.
