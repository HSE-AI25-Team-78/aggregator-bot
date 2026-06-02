# Targeted Weak Class Pipeline Report

- Seed labeled rows: **258**
- Capped selected rows: **2570**
- Weak-class targeted rows: **58**
- Augmented train rows: **2847**

## Weak-class additions by label

- Медицина: 27
- Общество, государство, политика: 24
- Развлечения: 5
- Общее: 2

## Metrics on original labeled holdout

- Baseline MultinomialNB F1_macro: **0.0575**
- Previous best exported MultinomialNB F1_macro: **0.2584**
- Targeted MultinomialNB F1_macro: **0.2741**
- Targeted LogisticRegression F1_macro: **0.1995**
- Service export updated: **yes**

## Targeted NB report

```text
                                 precision    recall  f1-score   support

          ИТ и телекоммуникации       0.00      0.00      0.00         2
           Искусство и культура       0.50      0.67      0.57         3
                        История       0.00      0.00      0.00         1
                       Медицина       0.40      1.00      0.57         2
                Наука и техника       0.67      0.50      0.57         4
                          Общее       0.00      0.00      0.00         5
Общество, государство, политика       0.25      0.22      0.24         9
                   Происшествия       1.00      0.86      0.92         7
                    Развлечения       0.00      0.00      0.00         2
                          Спорт       0.00      0.00      0.00         1
                      Экономика       0.09      0.33      0.14         3

                       accuracy                           0.38        39
                      macro avg       0.26      0.33      0.27        39
                   weighted avg       0.37      0.38      0.36        39

```