import pandas as pd
import json
import os

json_files = [
    "company_info.json",
    "income_statements.json",
    "balance_sheets.json",
    "cash_flows.json"
]

for fname in json_files:
    json_path = os.path.join("data", "financial", fname)
    csv_path = json_path.replace(".json", ".csv")
    if not os.path.exists(json_path):
        print(f"{json_path} not found, skipping.")
        continue
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"{csv_path} created!") 