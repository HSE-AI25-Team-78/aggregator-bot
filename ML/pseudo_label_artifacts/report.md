# Pseudo Label Pipeline Report

- Seed labeled rows: **258**
- External candidates: **6000**
- Accepted pseudo labels: **3750**
- Rejected candidates: **2250**
- Augmented train rows: **3969**

## Accepted by label

- ИТ и телекоммуникации: 750
- Экономика: 500
- Искусство и культура: 500
- Наука и техника: 500
- Происшествия: 250
- История: 250
- Развлечения: 250
- Медицина: 250
- Общество, государство, политика: 250
- Спорт: 250

## Metrics on original labeled holdout

- Baseline LogisticRegression F1_macro: **0.4200**
- Baseline MultinomialNB F1_macro: **0.0575**
- Previous best exported MultinomialNB F1_macro: **0.2399**
- Augmented LogisticRegression F1_macro: **0.2289**
- Augmented MultinomialNB F1_macro: **0.2584**
- Service export updated: **yes**

## Baseline NB report

```text
                                 precision    recall  f1-score   support

          ИТ и телекоммуникации       0.00      0.00      0.00         2
           Искусство и культура       0.00      0.00      0.00         3
                        История       0.00      0.00      0.00         1
                       Медицина       0.00      0.00      0.00         2
                Наука и техника       0.00      0.00      0.00         4
                          Общее       0.00      0.00      0.00         5
Общество, государство, политика       0.24      1.00      0.38         9
                   Происшествия       1.00      0.14      0.25         7
                    Развлечения       0.00      0.00      0.00         2
                          Спорт       0.00      0.00      0.00         1
                      Экономика       0.00      0.00      0.00         3

                       accuracy                           0.26        39
                      macro avg       0.11      0.10      0.06        39
                   weighted avg       0.23      0.26      0.13        39

```

## Augmented NB report

```text
                                 precision    recall  f1-score   support

          ИТ и телекоммуникации       0.29      1.00      0.44         2
           Искусство и культура       0.60      1.00      0.75         3
                        История       0.00      0.00      0.00         1
                       Медицина       0.00      0.00      0.00         2
                Наука и техника       0.43      0.75      0.55         4
                          Общее       0.00      0.00      0.00         5
Общество, государство, политика       0.00      0.00      0.00         9
                   Происшествия       1.00      0.57      0.73         7
                    Развлечения       0.00      0.00      0.00         2
                          Спорт       0.00      0.00      0.00         1
                      Экономика       0.23      1.00      0.38         3

                       accuracy                           0.38        39
                      macro avg       0.23      0.39      0.26        39
                   weighted avg       0.30      0.38      0.30        39

```