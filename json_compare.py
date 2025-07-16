import json
from difflib import SequenceMatcher

# Load the two JSON files
with open('/Users/macm3/Downloads/old_pollen_db.json') as f:
    old_schema = json.load(f)

with open('/Users/macm3/Downloads/new_pollen_db.json') as f:
    new_schema = json.load(f)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def compare_schemas(old_schema, new_schema):
    table_diffs = {}

    for table in old_schema:
        if table not in new_schema:
            table_diffs[table] = {"missing_in_new": True}
            continue

        old_cols = old_schema[table]
        new_cols = new_schema[table]

        diffs = {
            "columns_only_in_old": [],
            "columns_only_in_new": [],
            "type_mismatches": [],
            "possible_renamed_columns": [],
        }

        # Compare column names
        for col in old_cols:
            if col not in new_cols:
                diffs["columns_only_in_old"].append(col)
                # Try to find a similar column in new
                similar_cols = [(new_col, similar(col, new_col)) for new_col in new_cols]
                similar_cols = [c for c in similar_cols if c[1] > 0.8]
                if similar_cols:
                    best_match = max(similar_cols, key=lambda x: x[1])
                    diffs["possible_renamed_columns"].append((col, best_match[0]))
            else:
                # Check type
                old_type = old_cols[col][0]
                new_type = new_cols[col][0]
                if old_type != new_type:
                    diffs["type_mismatches"].append((col, old_type, new_type))

        # Columns in new but not in old
        for col in new_cols:
            if col not in old_cols:
                diffs["columns_only_in_new"].append(col)

        table_diffs[table] = diffs

    # Tables in new but not in old
    for table in new_schema:
        if table not in old_schema:
            table_diffs[table] = {"missing_in_old": True}

    return table_diffs

# Run the comparison
schema_diffs = compare_schemas(old_schema, new_schema)

# Display results for each table
for table, diffs in schema_diffs.items():
    print(f"\n=== Table: {table} ===")
    if "missing_in_new" in diffs:
        print("❌ Missing in new schema")
        continue
    if "missing_in_old" in diffs:
        print("❌ Missing in old schema")
        continue
    if diffs["columns_only_in_old"]:
        print("🔴 Columns only in old:", diffs["columns_only_in_old"])
    if diffs["columns_only_in_new"]:
        print("🟢 Columns only in new:", diffs["columns_only_in_new"])
    if diffs["type_mismatches"]:
        print("🟡 Type mismatches:")
        for col, old_type, new_type in diffs["type_mismatches"]:
            print(f"    {col}: old='{old_type}', new='{new_type}'")
    if diffs["possible_renamed_columns"]:
        print("🔁 Possibly renamed columns:")
        for old_col, new_col in diffs["possible_renamed_columns"]:
            print(f"    {old_col} → {new_col}")
