# Pseudo Label Pipeline Report

- Seed labeled rows: **258**
- External candidates: **11400**
- Accepted pseudo labels: **6000**
- Rejected candidates: **5400**
- Augmented train rows: **6219**

## Accepted by label

- ИТ и телекоммуникации: 1800
- Экономика: 1200
- Наука и техника: 1200
- Происшествия: 600
- Искусство и культура: 600
- Спорт: 600

## Metrics on original labeled holdout

- Baseline LogisticRegression F1_macro: **0.4200**
- Baseline MultinomialNB F1_macro: **0.0575**
- Augmented LogisticRegression F1_macro: **0.2177**
- Augmented MultinomialNB F1_macro: **0.2391**
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

          ИТ и телекоммуникации       0.25      1.00      0.40         2
           Искусство и культура       0.67      0.67      0.67         3
                        История       0.00      0.00      0.00         1
                       Медицина       0.00      0.00      0.00         2
                Наука и техника       0.40      0.50      0.44         4
                          Общее       0.00      0.00      0.00         5
Общество, государство, политика       0.00      0.00      0.00         9
                   Происшествия       1.00      0.71      0.83         7
                    Развлечения       0.00      0.00      0.00         2
                          Спорт       0.00      0.00      0.00         1
                      Экономика       0.17      1.00      0.29         3

                       accuracy                           0.36        39
                      macro avg       0.23      0.35      0.24        39
                   weighted avg       0.30      0.36      0.29        39

```