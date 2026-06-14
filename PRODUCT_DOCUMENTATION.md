# Aggregator Bot Product Documentation

## 1. Что это за продукт

`Aggregator Bot` — это система персонализированной новостной выдачи в Telegram.  
Система собирает публикации из набора Telegram-каналов, обрабатывает их, относит к тематическим категориям и выдаёт пользователю персональную ленту и дайджест.

В состав продукта входят:

- Telegram-бот для пользовательского взаимодействия;
- сервис API для классификации и диагностики;
- локальный dashboard для продуктовых и операционных метрик;
- pipeline обновления корпуса новостей;
- ML-контур классификации и ранжирования;
- событийный слой, который объединяет близкие публикации в сюжеты.

---

## 2. Назначение продукта

Назначение системы — формировать для пользователя персональную новостную выдачу на основе:

- выбранных тем;
- выбранных источников;
- поведенческих сигналов пользователя;
- актуального корпуса новостей.

Система должна решать следующие задачи:

- уменьшать объём нерелевантных публикаций;
- поддерживать поиск по новостям внутри интересующих тематик;
- группировать близкие публикации в один сюжет;
- давать пользователю инструменты управления своей лентой;
- обеспечивать регулярное обновление корпуса и контроль состояния сервиса.

Целевая единица выдачи в продукте — не отдельный пост, а сюжет, собранный из нескольких публикаций.

---

## 3. Высокоуровневая архитектура

### 3.1 Основные runtime-компоненты

1. `bot.telegram_bot`
   - Telegram-бот, который общается с пользователем.

2. `service.app`
   - FastAPI API для классификации, health-check и сервисного доступа.

3. `bot.live_dashboard`
   - локальный live-dashboard для продуктовой и операционной аналитики.

4. `bot.refresh_pipeline`
   - фоновый пайплайн обновления новостей.

5. `bot.recommender`
   - recommendation engine: загрузка корпуса, ранжирование, event-layer, профилирование.

6. `bot.event_builder`
   - построение событийного индекса из новостного корпуса.

7. `service.process`
   - сервисный inference layer, который грузит модель и запускает классификацию.

---

## 4. Карта репозитория

### 4.1 Ключевые папки

- `bot/`
  - логика бота, рекомендаций, dashboard, refresh pipeline

- `service/`
  - FastAPI API, DB-слой, inference-слой

- `ML/`
  - эксперименты, обучение, офлайн-оценка, feedback loop, MLflow-артефакты

- `data/`
  - основной локальный корпус новостей по базовым каналам

- `bot_data/`
  - runtime-данные бота:
    - профили
    - event log
    - импортированные кастомные каналы
    - refresh status
    - runtime logs
    - event index

- `scripts/autostart/`
  - автозапуск, старт, остановка процессов

### 4.2 Самые важные файлы

- `bot/telegram_bot.py`
- `bot/recommender.py`
- `bot/event_builder.py`
- `bot/refresh_pipeline.py`
- `bot/live_dashboard.py`
- `bot/analytics_report.py`
- `service/app.py`
- `service/process.py`
- `service/db.py`
- `service/config/model_manifest.json`
- `bot/config/base_channels.json`
- `bot/config/recommender_config.json`
- `scripts/autostart/start_all.ps1`
- `scripts/autostart/register_autostart.ps1`

---

## 5. Главные пользовательские возможности

### 5.1 Онбординг

При первом запуске бот собирает профиль пользователя:

- интересующие темы;
- предпочтительные источники;
- пользовательские Telegram-каналы.

### 5.2 Лента

Кнопка `📰 Моя лента` выдаёт персональную ленту:

- сначала по темам профиля;
- потом с учётом лайков, дизлайков и предпочтений;
- затем уже с учётом источников и кастомных каналов.

### 5.3 Дайджест

Кнопка `🗞 Дайджест` показывает компактную подборку:

- основные сюжеты;
- краткий формат;
- несколько источников по одному сюжету;
- короткая сводка.

### 5.4 Поиск

Кнопка `🔎 Найти новости` работает по ключевым словам, но учитывает темы профиля.

Пример:

- пользовательская тема: `Экономика`
- запрос: `биткоин`

Ожидаемая логика:

- искать по словам про `биткоин`;
- показывать результаты внутри экономической повестки.

### 5.5 Управление лентой

Пользователь может управлять лентой через карточки:

- `👍 Лайк`
- `👎 Неинтересно`
- `🔁 Похожее`
- `✨ Почему`
- `🚫 Источник / ♻️ Вернуть источник`
- `🙈 Тема / ♻️ Вернуть тему`
- `🧩 Ещё по сюжету`

Также есть экран:

- `☰ Ещё -> 🎛 Управление лентой`

где можно:

- увидеть скрытые темы;
- увидеть скрытые источники;
- вернуть их по одному;
- вернуть все фильтры сразу.

---

## 6. Event-based архитектура

### 6.1 Что было раньше

Раньше система была в основном `post-based`:

- ранжировались посты;
- потом они визуально группировались.

### 6.2 Что есть сейчас

Теперь есть отдельный слой событий:

- `EventCluster` в `bot/event_builder.py`

Событие хранит:

- `event_id`
- `topic`
- `title`
- `summary`
- `anchor_item_id`
- `item_ids`
- `sources`
- `source_count`
- `item_count`
- `published_from`
- `published_to`

### 6.3 Где хранится событийный индекс

- `bot_data/event_index.json`

Это runtime-индекс событий, который пересобирается recommender’ом.

### 6.4 Как строится событие

Событийный builder использует:

- тематическую метку;
- similarity между постами;
- временное окно;
- overlap токенов;
- anchor item как представитель сюжета.

### 6.5 Ограничения текущей реализации

Текущая реализация событийного слоя рабочая, но не окончательная.

На текущем этапе отсутствуют:

- более сильного кластеризатора;
- нормального multi-document summary;
- устойчивых event title generation;
- event-level ranking bonus за число независимых источников;
- более сильной дедупликации.

---

## 7. Recommendation engine

### 7.1 Что делает recommender

`bot/recommender.py`:

- загружает CSV-корпус;
- очищает тексты;
- применяет тематическую модель;
- строит TF-IDF представление;
- поддерживает similarity;
- строит event-index;
- выдаёт персональные рекомендации.

### 7.2 Основные режимы рекомендаций

- `latest_events`
- `recommend_events_for_topics`
- `recommend_events_for_profile`
- `recommend_events_for_query`
- `similar_events_to_item`

### 7.3 Персонализация

Персонализация учитывает:

- выбранные темы;
- выбранные источники;
- пользовательские каналы;
- лайки;
- дизлайки;
- скрытые темы;
- скрытые источники;
- similarity to liked content.

### 7.4 Ranking config

Файл:

- `bot/config/recommender_config.json`

Используется для настройки весов ранжирования без переписывания кода.

---

## 8. ML-модель и классификация

### 8.1 Канонический датасет и базовые размеры

Основной размеченный датасет для обучения лежит в:

- `ML/raw_posts_labeled.csv`

Это базовый 11-классовый датасет, от которого строятся:

- baseline-эксперименты;
- pseudo-label пайплайны;
- targeted weak-class пайплайны;
- topicality gate;
- offline recommendation evaluation.

Ключевые размеры, которые регулярно используются в контурах обучения:

- исходный seed-dataset после очистки: `258` строк;
- baseline train: `179` строк;
- baseline val: `39` строк;
- финальный test в ряде пайплайнов: `39` строк;
- train для topicality gate: `206` строк;
- test для topicality gate: `52` строки.

Практический смысл этого раздела:

- небольшой объём исходной ручной разметки был главным ограничением для качества классификатора;
- поэтому в проект были добавлены pseudo-label и targeted bootstrap этапы.

### 8.2 Целевые классы

Текущая схема классификации использует `11` тем:

- `ИТ и телекоммуникации`
- `Искусство и культура`
- `История`
- `Медицина`
- `Наука и техника`
- `Общее`
- `Общество, государство, политика`
- `Происшествия`
- `Развлечения`
- `Спорт`
- `Экономика`

Эти названия зафиксированы в deploy-manifest и используются одинаково:

- в API;
- в recommender;
- в event layer;
- в analytics;
- в offline evaluation.

### 8.3 Текущая deploy-модель

На текущий момент в продукте как основная модель используется:

- модель: `CalibratedLinearSVC`
- источник экспорта: `ML.train_stronger_deploy_model`
- задача: `news_topic_classification`

Боевая конфигурация зафиксирована в:

- [model_manifest.json](C:/Users/Yaroslav/Documents/aggregator-bot/service/config/model_manifest.json)

Текущие test-метрики из manifest:

- `f1_macro = 0.3943389943389944`
- `accuracy = 0.38461538461538464`
- `precision_macro = 0.44787485242030695`
- `recall_macro = 0.40043290043290036`

Артефакты deploy-контура:

- `service/config/model_manifest.json`
- `service/config/models/CalibratedLinearSVC.pkl`
- `service/config/vectorizer.pkl`
- `service/config/label_encoder.pkl`
- `service/config/topicality_gate.pkl`
- `service/config/topicality_gate_vectorizer.pkl`

Manifest нужен как единая точка правды для runtime-компонентов.  
И API, и бот читают оттуда:

- имя модели;
- набор классов;
- параметры векторизации;
- test-метрики;
- пути до артефактов;
- конфигурацию topicality gate.

### 8.4 Параметры векторизации и модели

Текущая deploy-векторизация:

- `TF-IDF`
- `ngram_range = (1, 2)`
- `min_df = 1`
- `max_features = 15000`
- `sublinear_tf = true`
- итоговый размер словаря: `15000`

Текущая deploy-модель обучается в `ML/train_stronger_deploy_model.py` и сравнивает несколько кандидатов.

Параметры кандидатов на этапе выбора более сильной модели:

- `LogisticRegression`
  - `max_iter = 4000`
  - `C = 2.0`
  - `class_weight = balanced`
- `SGDClassifier(loss="log_loss")`
  - `alpha = 1e-5`
  - `max_iter = 4000`
  - `class_weight = balanced`
  - `random_state = 42`
- `CalibratedLinearSVC`
  - базовый `LinearSVC(C = 1.0, class_weight = balanced)`
  - `CalibratedClassifierCV(cv = 3, method = "sigmoid")`

Почему в прод выбрана именно `CalibratedLinearSVC`:

- она показала лучший `f1_macro` среди проверенных кандидатов;
- калибровка даёт usable confidence, который затем используется в recommender и API.

### 8.5 Topicality gate перед multi-class классификацией

Сначала работает отдельный бинарный gate:

- `general / topical / uncertain`

Только после этого текст поступает в multi-class классификатор.

Текущая модель gate:

- модель: `LogisticRegression`
- задача: `topical_vs_general_gate`

Пороги gate:

- `general_threshold = 0.35`
- `topical_threshold = 0.55`

Метрики gate:

- `accuracy = 0.8846153846153846`
- `f1_binary = 0.9387755102040817`
- `precision_binary = 0.8846153846153846`
- `recall_binary = 1.0`

Дополнительные числа:

- topical rate на train: `0.8738`
- topical rate на test: `0.8846`
- средняя topical probability: `0.6966`

Практический эффект gate:

- часть шумных публикаций не форсируется в одну из 11 тем;
- класс `Общее` и состояние `Неуверенно` стали обрабатываться более честно;
- качество ленты и similarity-рекомендаций выросло после добавления gate.

### 8.6 Порог неуверенности

После topicality gate применяется дополнительный confidence threshold на выходе multi-class классификатора:

- `UNCERTAIN_CONFIDENCE_THRESHOLD = 0.24`

Если уверенность ниже этого порога, система помечает публикацию как `Неуверенно`.

Это влияет на:

- тематическую ленту;
- поиск;
- event layer;
- feedback loop;
- offline recommendation evaluation.

На текущем offline recommendation eval доля uncertain-предсказаний составляет:

- `uncertain_prediction_rate = 0.0115`

### 8.7 Эволюция ML-контура по этапам

Ниже перечислены основные этапы развития классификатора и их результаты.

#### 8.7.1 Baseline stage

Файл:

- `ML/artifacts/results/baseline/baseline_summary.json`

Размеры:

- `train_rows = 179`
- `val_rows = 39`
- `class_count = 11`

Параметры baseline-векторизации:

- `ngram_range = (1, 2)`
- `min_df = 5`
- `max_features = 20000`
- `vocabulary_size = 242`

Результаты baseline:

- `LinearSVC`: `f1_macro = 0.21474747474747471`, `accuracy = 0.4358974358974359`
- `LogisticRegression`: `f1_macro = 0.09501025290498974`
- `KNN_k5`: `f1_macro = 0.08577712609970675`
- `MultinomialNB`: `f1_macro = 0.07111281952719585`

Лучший baseline на этом этапе:

- `LinearSVC`

#### 8.7.2 Ранний финальный training pipeline

Файл:

- `ML/artifacts/results/final/training_summary.json`

Размеры:

- `train_rows = 179`
- `val_rows = 39`
- `test_rows = 39`
- `train_full_rows = 218`

Параметры:

- векторизация:
  - `ngram_range = (1, 2)`
  - `min_df = 1`
  - `max_features = 10000`
  - `sublinear_tf = true`
  - `vocabulary_size = 10000`
- `LogisticRegression`
  - `max_iter = 3000`
  - `C = 2.0`
- `MultinomialNB`
  - `alpha = 1.0`

Результаты:

- `LogisticRegression`: `f1_macro = 0.11794477545006087`
- `MultinomialNB`: `f1_macro = 0.05754352030947776`

Этот этап показал, что использовать `MultinomialNB` как основной deploy-классификатор нежелательно.

#### 8.7.3 Pseudo-label stage

Файл:

- `ML/pseudo_label_artifacts/summary.json`

Ключевые числа:

- `seed_rows = 258`
- `candidate_rows = 6000`
- `accepted_rows = 3750`
- `rejected_rows = 2250`
- `augmented_train_rows = 3969`

Пороги:

- `auto_accept_threshold = 0.53`
- `review_accept_threshold = 0.62`
- `margin_threshold = 0.1`

Результаты:

- `baseline_logreg_f1_macro = 0.42001332001332004`
- `baseline_nb_f1_macro = 0.05754352030947776`
- `previous_best_nb_f1_macro = 0.2584`
- `final_logreg_f1_macro = 0.2289371947125958`
- `final_nb_f1_macro = 0.25837924701561066`
- `service_export_updated = false`

Вывод по этапу:

- чистое масштабирование корпуса через auto-labeling дало ограниченный эффект;
- качество не выросло достаточно, чтобы автоматически обновить deploy-модель.

#### 8.7.4 Targeted weak-class stage

Файл:

- `ML/targeted_weak_class_artifacts/summary.json`

Ключевые числа:

- `seed_rows = 258`
- `selected_capped_rows = 2570`
- `weak_selected_rows = 58`
- `augmented_train_rows = 2847`

Результаты:

- `previous_best_nb_f1_macro = 0.2584`
- `baseline_nb_f1_macro = 0.05754352030947776`
- `final_nb_f1_macro = 0.27413762707880357`
- `final_logreg_f1_macro = 0.19953816237238697`
- `service_export_updated = true`

Смысл этапа:

- не просто увеличить размер train, а отдельно усилить слабые классы;
- этот этап стал последней удачной итерацией для ветки с `MultinomialNB`.

Правила отбора weak classes:

- `Общество, государство, политика`
  - `min_confidence = 0.34`
  - `min_margin = 0.17`
  - `require_proposed_match = true`
  - `limit = 220`
- `Медицина`
  - `min_confidence = 0.29`
  - `min_margin = 0.09`
  - `require_proposed_match = true`
  - `limit = 70`
- `История`
  - `min_confidence = 0.3`
  - `min_margin = 0.1`
  - `require_proposed_match = false`
  - `limit = 35`
- `Развлечения`
  - `min_confidence = 0.36`
  - `min_margin = 0.12`
  - `require_proposed_match = true`
  - `limit = 80`
- `Общее`
  - `min_confidence = 0.2`
  - `min_margin = 0.04`
  - `require_proposed_match = false`
  - `limit = 20`

#### 8.7.5 Переход на более сильную deploy-модель

Файлы:

- `ML/train_stronger_deploy_model.py`
- `ML/stronger_deploy_model_artifacts/summary.json`
- `ML/stronger_deploy_model_artifacts/report.md`

Ключевые числа:

- `seed_rows = 258`
- `augmented_train_rows = 2847`
- `test_rows = 39`
- `previous_best_nb_f1_macro = 0.2584`

Сравнивались:

- `CalibratedLinearSVC`
- `SGDClassifier(log_loss)`
- `LogisticRegression`

Итог:

- `CalibratedLinearSVC`: `f1_macro = 0.3943389943389944`
- `SGDClassifier_log_loss`: `f1_macro = 0.29668109668109666`
- `LogisticRegression`: `f1_macro = 0.19953816237238697`

Именно этот этап сформировал текущую production-конфигурацию.

#### 8.7.6 Слабые места текущей deploy-модели по классам

По отчёту лучшей модели видно, что качество по классам неравномерно.

Лучше всего на test-срезе классифицируются:

- `Спорт`
- `Происшествия`
- `ИТ и телекоммуникации`

Слабее всего выглядят:

- `История`
- `Общее`
- `Развлечения`
- `Экономика`

Поэтому `f1_macro = 0.3943` не означает, что все 11 тем распознаются одинаково хорошо.  
В продуктовой логике это компенсируется:

- topicality gate;
- uncertain threshold;
- topic-aware ranking;
- event-level grouping;
- фильтрацией по профилю пользователя.

### 8.8 Offline-оценка recommendation layer

Для продукта важны не только model-level метрики классификатора, но и метрики recommendation layer.

Файлы:

- `ML/offline_recommendation_eval.py`
- `ML/recommendation_eval_artifacts/summary.json`
- `ML/recommendation_eval_artifacts/report.md`

Оценка считается на `ML/raw_posts_labeled.csv` как offline proxy для двух сценариев:

- `recommend_events_for_topics`
- `similar_events_to_item`

Текущие значения:

- `topic_feed_precision@5 = 0.9636`
- `similarity_precision@5 = 0.8604`
- `topic_coverage = 1.0`
- `uncertain_prediction_rate = 0.0115`
- `topics_evaluated = 11`
- `k = 5`

Per-topic `topic_feed_precision@5`:

- `Общее = 1.0`
- `Наука и техника = 1.0`
- `ИТ и телекоммуникации = 1.0`
- `Общество, государство, политика = 1.0`
- `Экономика = 0.8`
- `Медицина = 1.0`
- `Искусство и культура = 1.0`
- `Развлечения = 1.0`
- `Спорт = 1.0`
- `История = 0.8`
- `Происшествия = 1.0`

Per-topic `similarity_precision@5`:

- `Общее = 0.84`
- `Наука и техника = 0.9333`
- `ИТ и телекоммуникации = 0.9167`
- `Общество, государство, политика = 0.88`
- `Экономика = 0.8667`
- `Медицина = 0.9286`
- `Искусство и культура = 0.9333`
- `Развлечения = 0.8667`
- `Спорт = 0.8857`
- `История = 0.48`
- `Происшествия = 0.9333`

Практический смысл этих метрик:

- topic feed в текущем состоянии работает заметно сильнее, чем ранние версии проекта;
- similarity-рекомендации тоже находятся на рабочем уровне;
- худшим направлением остаётся исторический контент.

### 8.9 Feedback loop и дообучение по пользовательским сигналам

Файлы:

- `ML/build_feedback_training_candidates.py`
- `ML/feedback_retrain_pipeline.py`
- `ML/feedback_loop_artifacts/feedback_retrain_summary.json`

Текущие числа:

- `seed_rows = 258`
- `feedback_rows = 1`
- `replicated_feedback_rows = 2`
- `augmented_train_rows = 221`
- `baseline_logreg_f1_macro = 0.42001332001332004`
- `baseline_nb_f1_macro = 0.05754352030947776`
- `previous_best_nb_f1_macro = 0.2584`
- `final_logreg_f1_macro = 0.4404329004329004`
- `final_nb_f1_macro = 0.05754352030947776`
- `service_export_updated = false`

Вывод:

- контур feedback loop уже встроен;
- слабые сигналы из пользовательского поведения уже собираются;
- объём обратной связи пока слишком мал, чтобы безопасно обновлять production-export автоматически.

### 8.10 MLflow и воспроизводимость

В проекте используется локальный MLflow-tracking.

Основное хранилище:

- `ML/mlruns/`

Покрываемые эксперименты:

- baseline experiments;
- финальный training;
- pseudo-label pipeline;
- targeted weak-class pipeline;
- stronger deploy model search;
- topicality gate training;
- feedback retrain experiments.

MLflow в этом проекте нужен не как внешний сервис, а как локальный реестр экспериментов:

- какие параметры использовались;
- какие метрики получились;
- какой run экспортировал текущую deploy-модель;
- какие артефакты были сохранены.

### 8.11 Почему model-level метрики нельзя читать отдельно от продукта

Классификация участвует не только в API-ручке `/forward`, но и в:

- ленте;
- поиске;
- дайджесте;
- event layer;
- аналитике;
- feedback loop.

Поэтому в системе одновременно важны два слоя качества.

1. `Model-level quality`
   - `f1_macro`
   - `accuracy`
   - `precision_macro`
   - `recall_macro`

2. `Recommendation-level quality`
   - `topic_feed_precision@5`
   - `similarity_precision@5`
   - `topic_coverage`
   - `uncertain_prediction_rate`
   - реальные product events из `bot_data/events.jsonl`

Именно поэтому в продуктовой оценке нельзя ориентироваться только на одно число `f1_macro`.  
В этом проекте решение о пригодности модели для runtime принимается по совокупности:

- качества multi-class классификации;
- стабильности topicality gate;
- поведения uncertainty policy;
- качества event-based recommendation layer.

---

## 9. FastAPI service

### 9.1 Назначение

Сервис нужен для:

- health-check;
- ready-check;
- model-info;
- prediction API;
- истории запросов;
- сервисной диагностики.

### 9.2 Основные эндпоинты

- `GET /`
- `GET /health`
- `GET /ready`
- `GET /model-info`
- `POST /forward`
- `POST /login`

### 9.3 Где код сервиса

- `service/app.py`
- `service/process.py`
- `service/db.py`
- `service/api_models.py`

### 9.4 База

Сейчас сервис использует SQLite:

- `aggregator_bot.db`

Это подходит для локальной эксплуатации и демо, но не является финальной production DB-архитектурой.

---

## 10. Корпус новостей

### 10.1 Базовые источники

Список базовых каналов управляется из:

- `bot/config/base_channels.json`

Это важно, потому что:

- корпус больше не зависит только от тех CSV, которые уже когда-то лежали в `data/`;
- новые базовые каналы можно добавлять системно.

### 10.2 Кастомные каналы

Пользователь может добавить свои источники:

- они импортируются в `bot_data/imported_channels/`
- участвуют в персональной выдаче

### 10.3 Структура данных

Основной корпус:

- `data/*.csv`

Кастомный пользовательский корпус:

- `bot_data/imported_channels/*.csv`

### 10.4 Политика хранения

Сейчас работает retention policy:

- быстрые новостные каналы: `14` дней
- экономика / IT / спорт / аналитика: `30` дней
- наука / история / культура / медицина: `90` дней
- кастомные каналы пользователя: `30` дней

Это реализовано в:

- `bot/refresh_pipeline.py`

---

## 11. Refresh pipeline

### 11.1 Назначение

Refresh pipeline:

- обновляет каналы;
- подтягивает новые посты;
- чистит старые посты по retention policy;
- пишет summary в status-файл.

### 11.2 Файл

- `bot/refresh_pipeline.py`

### 11.3 Status file

- `bot_data/refresh_status.json`

### 11.4 Что в нём хранится

- когда закончился последний refresh;
- сколько каналов обновилось;
- сколько пропущено;
- сколько ошибок;
- сколько строк осталось после retention.

### 11.5 Уже существующий планировщик

В системе уже был настроен refresh-task:

- `AggregatorBotNewsRefresh`

Он отвечает именно за обновление корпуса.

---

## 12. Live dashboard

### 12.1 URL

- `http://127.0.0.1:8765`

### 12.2 Назначение

Dashboard используется для наблюдения за состоянием продукта и runtime-компонентов.  
На панели отображаются:

- пользовательские метрики;
- метрики рекомендаций;
- состояние feedback loop;
- состояние модели;
- состояние refresh pipeline;
- здоровье корпуса (`Corpus Health`);
- окна retention;
- крупнейшие источники;
- метрики онбординга;
- DAU / WAU / MAU;
- поисковые запросы;
- разрезы показов, лайков и дизлайков.

### 12.3 Основные файлы

- `bot/live_dashboard.py`
- `bot/analytics_report.py`
- `bot/build_dashboard.py`

### 12.4 JSON-слой

- `http://127.0.0.1:8765/analytics.json`

---

## 13. Product analytics

### 13.1 Где хранятся события

- `bot_data/events.jsonl`

### 13.2 Какие события логируются

Примеры:

- `feed_shown`
- `search_query`
- `feedback_like`
- `feedback_dislike`
- `open_similar`
- `open_why`
- `mute_source`
- `unmute_source`
- `mute_topic`
- `unmute_topic`
- `onboarding_started`
- `onboarding_completed`
- `profile_view`
- `feed_controls_view`

### 13.3 Зачем это нужно

Эти события нужны для:

- dashboard;
- product analytics;
- feedback loop;
- оценки поведения пользователя.

---

## 14. Feedback loop

### 14.1 Что уже есть

В проекте уже есть первый feedback loop:

- сбор кандидатов из пользовательского поведения;
- retrain-эксперименты;
- сравнение с текущей моделью;
- safe-update только если модель реально лучше.

### 14.2 Основные файлы

- `ML/build_feedback_training_candidates.py`
- `ML/feedback_retrain_pipeline.py`
- `ML/feedback_loop_artifacts/`

### 14.3 Ограничение

Текущий feedback loop не является полностью автоматическим контуром самообучения.  
Сейчас это экспериментальный контур с контролем качества: модель обновляется только после отдельного прогона и сравнения результатов.

---

## 15. Автозапуск и стартовые скрипты

### 15.1 Что добавлено

Создан набор скриптов:

- `scripts/autostart/common.ps1`
- `scripts/autostart/start_bot.ps1`
- `scripts/autostart/start_dashboard.ps1`
- `scripts/autostart/start_api.ps1`
- `scripts/autostart/start_all.ps1`
- `scripts/autostart/stop_all.ps1`
- `scripts/autostart/register_autostart.ps1`

### 15.2 Что запускается

При полном старте поднимаются:

1. Telegram bot
2. Live dashboard
3. FastAPI service API

### 15.3 Где хранится автозапуск

Автозапуск регистрируется не через Task Scheduler, а через Startup folder текущего пользователя:

- `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\AggregatorBotSuite.cmd`

Это решение было выбрано потому, что в текущей среде создание новых scheduled tasks дало `Access denied`.

### 15.4 Runtime logs

Логи процессов лежат в:

- `bot_data/runtime_logs/telegram_bot.log`
- `bot_data/runtime_logs/telegram_bot.err.log`
- `bot_data/runtime_logs/live_dashboard.log`
- `bot_data/runtime_logs/live_dashboard.err.log`
- `bot_data/runtime_logs/service_api.log`
- `bot_data/runtime_logs/service_api.err.log`

---

## 16. Как стартовать систему вручную

### 16.1 Полный старт

```powershell
& .\scripts\autostart\start_all.ps1
```

### 16.2 Полная остановка

```powershell
& .\scripts\autostart\stop_all.ps1
```

### 16.3 Запустить только бота

```powershell
& .\scripts\autostart\start_bot.ps1
```

### 16.4 Запустить только dashboard

```powershell
& .\scripts\autostart\start_dashboard.ps1
```

### 16.5 Запустить только API

```powershell
& .\scripts\autostart\start_api.ps1
```

### 16.6 Установить автозапуск

```powershell
& .\scripts\autostart\register_autostart.ps1
```

---

## 17. Порты и endpoints

### 17.1 Dashboard

- URL: `http://127.0.0.1:8765`

### 17.2 FastAPI service

- Health: `http://127.0.0.1:8000/health`
- Ready: `http://127.0.0.1:8000/ready`
- Model info: `http://127.0.0.1:8000/model-info`

### 17.3 Telegram bot

Работает через polling и не использует внешний web-port.

---

## 18. Переменные окружения

Проект опирается на `.env`.

Наиболее важные переменные:

- `TELEGRAM_BOT_TOKEN`
- `API_ID`
- `API_HASH`
- `SESSION_NAME`
- `SECRET_KEY`
- `ALGORITHM`
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `DELETE_TOKEN`

Дополнительно для администратора могут использоваться:

- `ADMIN_USERNAME`
- `ADMIN_EMAIL`
- `ADMIN_FULL_NAME`
- `ADMIN_PASSWORD`

---

## 19. Текущее состояние продукта

На момент подготовки документации в системе реализованы:

- персональный onboarding;
- персональная лента;
- дайджест;
- поиск;
- лайки/дизлайки;
- скрытие и возврат тем/источников;
- live dashboard;
- refresh pipeline;
- retention policy;
- event index;
- event-first recommendation layer;
- FastAPI health endpoints;
- автозапуск через Startup folder текущего пользователя.

---

## 20. Ограничения текущей версии

### 20.1 Event layer ещё не финальный

Сейчас event layer уже отдельный архитектурный слой, но пока:

- summarization эвристическая;
- clustering ещё можно усиливать;
- ranking событий ещё можно делать умнее;
- event-title ещё не является идеальным multi-source title.

### 20.2 Модель классификации всё ещё можно улучшать

Несмотря на серьёзный прогресс, классификатор не идеален.  
Это влияет на:

- баланс тем;
- точность отдельных сюжетов;
- поисковую релевантность.

### 20.3 Хранение пока локальное

Сейчас корпус и события лежат локально в CSV/JSON.  
Для настоящего production-scale понадобилось бы:

- более структурированное хранилище;
- event store;
- централизованный DB-слой;
- возможно очередь задач.

---

## 21. Рекомендуемый порядок изучения проекта для нового человека

Если новый разработчик хочет понять систему быстро, лучше читать в таком порядке:

1. `PRODUCT_DOCUMENTATION.md`
2. `bot/telegram_bot.py`
3. `bot/recommender.py`
4. `bot/event_builder.py`
5. `service/app.py`
6. `service/process.py`
7. `bot/refresh_pipeline.py`
8. `bot/live_dashboard.py`
9. `bot/analytics_report.py`
10. `service/config/model_manifest.json`

---

## 22. Диагностика и troubleshooting

### 22.1 Если не отвечает dashboard

Проверить:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765/
```

Если ошибка:

- проверить процесс `bot.live_dashboard`
- посмотреть `bot_data/runtime_logs/live_dashboard.err.log`

### 22.2 Если не отвечает API

Проверить:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

Если ошибка:

- проверить процесс `uvicorn app:app`
- посмотреть `bot_data/runtime_logs/service_api.err.log`

### 22.3 Если бот не отвечает

Проверить:

- процесс `bot.telegram_bot`
- `bot_data/runtime_logs/telegram_bot.err.log`
- `TELEGRAM_BOT_TOKEN`

### 22.4 Если refresh не обновляет корпус

Проверить:

- `.env` содержит `API_ID` и `API_HASH`
- client session авторизована
- `bot_data/refresh_status.json`
- `bot_data/refresh_cron.log`
- `bot_data/refresh_cron.err.log`

### 22.5 Если event layer долго стартует

Это ожидаемо при большом корпусе, потому что:

- recommender заново читает корпус;
- строит TF-IDF;
- строит similarity-слой;
- строит event-index.

Именно поэтому после рестарта нужно давать системе время на инициализацию.

---

## 23. Направления дальнейшего развития

Приоритетные дальнейшие шаги:

1. Улучшение кластеризации событий
2. Добавление event-level бонуса за разнообразие источников
3. Улучшение сводки по событиям
4. Балансировка тем в выдаче
5. Вынос хранения событий в более структурированный слой
6. Улучшение мониторинга перекоса тем и дрейфа рекомендаций

---

## 24. Краткий operational checklist

Если нужно быстро проверить, что всё живо:

1. Бот запущен
2. Dashboard отвечает на `127.0.0.1:8765`
3. API отвечает на `127.0.0.1:8000/health`
4. `refresh_status.json` свежий
5. `event_index.json` существует
6. `runtime_logs/*.err.log` пустые или без критических ошибок

---

## 25. Итоговое состояние системы

На текущем этапе проект представляет собой набор согласованных компонентов:

- ingestion новостей из Telegram-каналов;
- ML-классификация текстов по темам;
- recommendation layer для персональной выдачи;
- event layer для группировки публикаций в сюжеты;
- dashboard для продуктового и операционного наблюдения;
- refresh pipeline с политикой хранения корпуса;
- scripts для запуска и автозапуска runtime-компонентов.

В таком виде система пригодна для локальной эксплуатации, демонстрации и дальнейшего развития как продуктового прототипа.
