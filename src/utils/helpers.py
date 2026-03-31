import json
from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, output_path: Path, index: bool = False) -> None:
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=index)


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def summarize_target(y: pd.Series) -> pd.DataFrame:
    summary = (
        y.value_counts(dropna=False)
         .rename_axis("label")
         .reset_index(name="count")
    )
    summary["share"] = summary["count"] / summary["count"].sum()
    return summary