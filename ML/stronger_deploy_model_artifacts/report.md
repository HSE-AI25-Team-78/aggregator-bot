# Stronger Deploy Model Report

- Seed rows: **258**
- Augmented train rows: **2847**
- Test rows: **39**
- Previous best NB F1_macro: **0.2584**
- Best model: **CalibratedLinearSVC**
- Best F1_macro: **0.3943**
- Service export updated: **yes**

## Results

- CalibratedLinearSVC: f1_macro=0.3943, accuracy=0.3846, precision_macro=0.4479, recall_macro=0.4004
- SGDClassifier_log_loss: f1_macro=0.2967, accuracy=0.4103, precision_macro=0.2983, recall_macro=0.3229
- LogisticRegression: f1_macro=0.1995, accuracy=0.2564, precision_macro=0.2825, recall_macro=0.2211

## Best model report

```text
                                 precision    recall  f1-score   support

          ИТ и телекоммуникации       1.00      0.50      0.67         2
           Искусство и культура       0.29      0.67      0.40         3
                        История       0.00      0.00      0.00         1
                       Медицина       0.50      0.50      0.50         2
                Наука и техника       0.50      0.50      0.50         4
                          Общее       0.00      0.00      0.00         5
Общество, государство, политика       0.75      0.33      0.46         9
                   Происшествия       0.80      0.57      0.67         7
                    Развлечения       0.00      0.00      0.00         2
                          Спорт       1.00      1.00      1.00         1
                      Экономика       0.09      0.33      0.14         3

                       accuracy                           0.38        39
                      macro avg       0.45      0.40      0.39        39
                   weighted avg       0.50      0.38      0.40        39

```