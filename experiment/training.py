import json
import pandas as pd
import uuid

# path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/output/cross_mapping/time-chf_gissi-hf_gissi-hf_outcomes_sapbert_concept-only.json"

# with open(path) as f:
#     raw = json.load(f)

# keep_status = {"Complete Match (Identical)", "Complete Match (Compatible)"}

# rows = []

# for src_var, bundle in raw.items():
#     mappings = bundle.get("mappings", [])
#     source_study = bundle.get("from")
#     for m in mappings:
#         status = m.get("harmonization_status")
#         if status not in keep_status:
#             continue

#         target_study = m.get("target_study")
#         if not target_study:
#             continue

#         # dynamic keys, e.g. "gissi-hf_target", "gissi-hf_outcomes_tlabel"
#         t_var_key = f"{target_study}_target"
#         t_label_key = f"{target_study}_tlabel"
#         mapping_type = m.get(f"{target_study}_mapping_type")
#         rows.append({
#             "source_study": source_study,
#             "target_study": target_study,
#             "variable_name_source": m.get("s_source"),
#             "variable_name_target": m.get(t_var_key),
#             "variable_label_source": m.get("s_slabel"),
#             "variable_label_target": m.get(t_label_key),
#             "mapping_type": mapping_type,
#             "harmonization_status": status,
            
#         })

# gold_pairs = pd.DataFrame(rows)

# # 🔁 Standardize column names
# gold_pairs = gold_pairs.rename(columns={
#     "source_study": "source_study_id",
#     "target_study": "target_study_id",
#     "variable_name_source": "source_variable_id",
#     "variable_name_target": "target_variable_id",
#     "variable_label_source": "source_variable_label",
#     "variable_label_target": "target_variable_label",
#     "harmonization_status": "mapping_status",
# })

# # (Optional but nice) add a stable mapping_id
# gold_pairs.insert(
#     0,  # at first position
#     "mapping_id",
#     [str(uuid.uuid4()) for _ in range(len(gold_pairs))]
# )

# print(gold_pairs.head())
# print(len(gold_pairs), "pairs kept")

# out_path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/gold_pairs_1_fair.csv"
# gold_pairs.to_csv(out_path, index=False)
# print("Saved to:", out_path)


# merge two gold pair files
# file1 = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/gold_pairs_1_fair.csv"
# file2 = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/gold_pairs_2_fair.csv"
# df1 = pd.read_csv(file1)
# df2 = pd.read_csv(file2)
# merged = pd.concat([df1, df2], ignore_index=True)
# # drop duplicates
# merged = merged.drop_duplicates(subset=["source_study_id", "target_study_id", "source_variable_id", "target_variable_id"])
# print(f"Merged dataframe has {len(merged)} unique pairs.")
# merged.to_csv("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/gold_pairs_combined_fair.csv", index=False)
# print("Merged file saved.")