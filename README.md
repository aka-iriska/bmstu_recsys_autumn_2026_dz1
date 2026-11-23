# Рекомендательная система: Kaggle «Dz 1: Recommender System» (2026)

Задача: по логам музыкального сервиса (прослушивания, лайки, дизлайки) выдать каждому пользователю из `test_users.csv` до 10 треков, которые он **будет слушать после границы train**. Повторные прослушивания засчитываются. 

Метрика - NDCG@10, усреднённая по пользователям.

Решение двухстадийное:

1. **Отбор кандидатов.** Из ~880 тыс. треков для каждого пользователя оставляем несколько сотен правдоподобных: 
не больше ~550 (до 300 из истории, 100 от iALS, до 150 популярных: по 50 за 1, 7 и 30 дней; i2i выключен), 
на деле меньше из-за пересечений источников.
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

- `validate` считает локальный NDCG@10, подбирает число деревьев и сохраняет в `output/` кандидатов двух срезов (`fold_a.parquet`, `fold_b.parquet`), правильные ответы (`truth.parquet`) и список популярного (`pop_items.json`).
- `validate --ranker-only` делает то же без пересборки кандидатов, по сохранённым файлам: около 1,5 минуты. Подходит для проверки параметров ранкера.
- `tune` подбирает параметры LightGBM через Optuna по сохранённым кандидатам и пишет `output/best_params.json`; 
Чтобы вернуться к параметрам по умолчанию, удалить `best_params.json`. Запускать `tune` нужно после того, как набор признаков окончательный: параметры подобраны под конкретную таблицу.
- `submit` обучает финальный ранкер и пишет `output/submission.csv`.

Параметры командной строки: `--rounds N` (максимум деревьев при validate), `--final-rounds N` (число деревьев при submit), `--ranker-only`, `--trials N` (число испытаний в tune).
Пути переопределяются переменными окружения `RECSYS_DATA` и `RECSYS_OUT`.

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
| `src/s2_candidates/sources/pop.py` | 2 | `PopSource`: популярное за 1, 7 и 30 дней |
| `src/s2_candidates/sources/i2i.py` | 2 | `I2ISource`: item2item по совместной встречаемости в сессиях (сейчас выключен в `SOURCES`) |
| `src/s2_candidates/sources/als.py` | 2 | `AlsSource`: iALS через `implicit`, веса уверенности, кандидаты, признаки |
| `src/s3_ranker/ranker.py` | 3 | `train_ranker`, `recommend` (LightGBM LambdaRank) |
| `src/s3_ranker/tuning.py` | 3 | `tune_ranker`: подбор параметров LightGBM через Optuna |
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
- Сдача: выбрать 2 сабмита для Private по **локальной** валидации, а не по Public.
