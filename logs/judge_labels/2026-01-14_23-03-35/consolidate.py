"""Consolidate judge label JSON files into Parquet format."""
import json
from pathlib import Path
import polars as pl

# Get all persona label files
output_dir = Path(__file__).parent
json_files = sorted(output_dir.glob("persona_*_labels.json"))

# Collect all labels
all_rows = []
for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    persona_id = data["persona_id"]
    for label in data["labels"]:
        scores = label["scores"]
        row = {
            "persona_id": persona_id,
            "t_index": label["t_index"],
            "date": label["date"],
            "alignment_vector": [
                scores["self_direction"],
                scores["stimulation"],
                scores["hedonism"],
                scores["achievement"],
                scores["power"],
                scores["security"],
                scores["conformity"],
                scores["tradition"],
                scores["benevolence"],
                scores["universalism"],
            ],
            "alignment_self_direction": scores["self_direction"],
            "alignment_stimulation": scores["stimulation"],
            "alignment_hedonism": scores["hedonism"],
            "alignment_achievement": scores["achievement"],
            "alignment_power": scores["power"],
            "alignment_security": scores["security"],
            "alignment_conformity": scores["conformity"],
            "alignment_tradition": scores["tradition"],
            "alignment_benevolence": scores["benevolence"],
            "alignment_universalism": scores["universalism"],
        }
        all_rows.append(row)

# Create DataFrame and write to Parquet
df = pl.DataFrame(all_rows)
df.write_parquet(output_dir / "judge_labels.parquet")

print(f"Consolidated {len(all_rows)} entries from {len(json_files)} personas")
print(f"Output: {output_dir / 'judge_labels.parquet'}")

# Print score distribution
print("\nScore Distribution:")
value_cols = [c for c in df.columns if c.startswith("alignment_") and c != "alignment_vector"]
for col in value_cols:
    counts = df[col].value_counts().sort("count", descending=True)
    value_name = col.replace("alignment_", "")
    neg = df.filter(pl.col(col) == -1).height
    zero = df.filter(pl.col(col) == 0).height
    pos = df.filter(pl.col(col) == 1).height
    print(f"  {value_name:15s}: -1={neg:2d}, 0={zero:2d}, +1={pos:2d}")
