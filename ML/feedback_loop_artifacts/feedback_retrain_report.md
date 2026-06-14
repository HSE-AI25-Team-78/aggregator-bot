# Feedback Retrain Report

- Seed rows: **258**
- Accepted feedback rows: **1**
- Replicated feedback rows: **2**
- Augmented train rows: **221**

## Feedback by label

- Общество, государство, политика: 1

## Metrics on original labeled holdout

- Baseline MultinomialNB F1_macro: **0.0575**
- Previous best exported MultinomialNB F1_macro: **0.2584**
- Feedback-augmented MultinomialNB F1_macro: **0.0575**
- Service export updated: **no**

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

## Feedback-augmented NB report

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