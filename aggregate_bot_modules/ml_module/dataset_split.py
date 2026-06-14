from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from .paths import PROCESSED_PATH, SPLIT_DIR, TEST_PATH, TRAIN_PATH, VAL_PATH, ensure_artifact_dirs


def main(
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> None:
    """
    Делим processed_posts.csv на train / val / test и сохраняем в data/splits.
    """

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Не найден файл {PROCESSED_PATH}")

    print(f"[*] Загружаем обработанный датасет: {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH)

    if "topic" not in df.columns:
        raise ValueError("В processed_posts.csv нет колонки 'topic'")

    df = df.dropna(subset=["topic"])
    df["topic"] = df["topic"].astype(str)

    print(f"[*] Всего строк для разбиения: {len(df)}")

    # === 1. Выделяем test ===
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["topic"],
        random_state=random_state,
    )

    # === 2. Делим train_val на train и val ===
    val_rel_size = val_size / (1.0 - test_size)

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_rel_size,
        stratify=df_train_val["topic"],
        random_state=random_state,
    )

    print(f"[*] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Создаём папку splits/
    ensure_artifact_dirs()

    # Сохраняем
    df_train.to_csv(TRAIN_PATH, index=False)
    df_val.to_csv(VAL_PATH, index=False)
    df_test.to_csv(TEST_PATH, index=False)

    print(f"[+] Train сохранён в: {TRAIN_PATH}")
    print(f"[+] Val   сохранён в: {VAL_PATH}")
    print(f"[+] Test  сохранён в: {TEST_PATH}")


if __name__ == "__main__":
    main()
