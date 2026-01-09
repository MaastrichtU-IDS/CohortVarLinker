
import os
import glob
import time
import pandas as pd
from src.utils import load_dictionary


def normalize_id(series):
    """Clean OMOP IDs (remove .0, whitespace, NaN)."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": "", "None": ""})
    )


def normalize_unit(unit_str):
    """Normalize UCUM-like units: remove ucum:, braces, and lowercase."""
    if not isinstance(unit_str, str):
        return ""
    unit_str = unit_str.strip().lower()
    unit_str = unit_str.replace("ucum:", "")
    unit_str = unit_str.replace("{counts}", "")
    unit_str = unit_str.replace(" ", "")
    unit_str = unit_str.replace("[", "").replace("]", "")
    return unit_str

def unit_in_row(row_units, search_unit):
    """Return True if the search unit matches any pipe-separated unit in the row."""
    if not isinstance(row_units, str):
        return False
    row_units = [u.strip().lower() for u in row_units.split('|')]
    return search_unit.lower() in row_units


def explore_units_for_each_Cohort(dir_path, recreate=False):
    # preferred_units_path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/S7-preferred_units-ABC.csv"
    standard_conversion_path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/units_conversion.csv"

    # preferred_units = load_dictionary(preferred_units_path)
    # preferred_units.columns = [col.lower().strip() for col in preferred_units.columns]

    standard_conversion = load_dictionary(standard_conversion_path)
    standard_conversion.columns = [col.lower().strip() for col in standard_conversion.columns]

    # Normalize OMOP IDs and units
    standard_conversion["variable omop id"] = normalize_id(standard_conversion["variable omop id"])
    # standard_conversion["omop id"] = normalize_id(standard_conversion["omop id"])
    standard_conversion["conventional units"] = standard_conversion["conventional units"].astype(str).map(normalize_unit)
    standard_conversion["si units"] = standard_conversion["si units"].astype(str).map(normalize_unit)

    results = []
    print(f"✅ Normalized preferred_units and conversion tables")

    start_time = time.time()

    for cohort_folder in os.listdir(dir_path):
        if cohort_folder.startswith('.'):
            continue

        cohort_path = os.path.join(dir_path, cohort_folder)
        if not os.path.isdir(cohort_path):
            continue

        # find metadata file
        patterns = ("*.csv", "*.xlsx", "*.json")
        file_candidates = []
        for pat in patterns:
            file_candidates.extend(glob.glob(os.path.join(cohort_path, pat)))

        cohort_metadata_file = None
        for file in file_candidates:
            if file.lower().endswith((".csv", ".xlsx")):
                cohort_metadata_file = file

        if not cohort_metadata_file:
            continue

        df = load_dictionary(cohort_metadata_file)
        df.columns = [col.lower().strip() for col in df.columns]

        df = df[df["domain"].str.lower() == "measurement"]
        df = df[df["unit concept name"].notna() & (df["unit concept name"].str.strip() != "")]

        # Normalize OMOP IDs and units in metadata
        df["variable omop id"] = normalize_id(df["variable omop id"])
        df["unit concept code"] = df["unit concept code"].astype(str).map(normalize_unit)

        for _, row in df.iterrows():
            var_name = row.get("variable concept name", "")
            var_id = row.get("variable omop id", "")
            visit_name = row.get("visit concept name", "")
            unit = row.get("unit concept code", "")

            preferred_unit = ""
            in_preferred = "No (Not in preferred list)"
            convertible = "No"
            si_unit = ""
            conversion_formula = ""

            # Step 1: check preferred units
            match = standard_conversion[standard_conversion["variable omop id"] == var_id]
            if not match.empty:
                preferred_unit = normalize_unit(match["preferred unit code"].values[0])
                if unit == preferred_unit:
                    in_preferred = "Yes"

            # Step 2: if not preferred, check conversion file
            if in_preferred != "Yes":
                std_match = standard_conversion[standard_conversion["variable omop id"] == var_id]

                # Relaxed match: try also by unit presence if OMOP ID row doesn't exist
                if std_match.empty:
                    mask_conv = standard_conversion["conventional units"].apply(unit_in_row, search_unit=unit)
                    mask_si = standard_conversion["si units"].apply(unit_in_row, search_unit=unit)
                    std_match = standard_conversion[mask_conv | mask_si]

                if not std_match.empty:
                    convertible = "Yes"
                    si_unit = std_match["si units"].values[0]
                    multiplier = std_match["conv to si"].values[0]

                    try:
                        multiplier_val = float(multiplier)
                        conversion_formula = f"value * {multiplier_val}"
                    except Exception:
                        conversion_formula = "NA"

            results.append({
                "Cohort": cohort_folder,
                "Measurement (Variable Concept Name)": var_name,
                "OMOP ID": var_id,
                "Visit Concept Name": visit_name,
                "Unit": unit,
                "In Preferred Unit (Yes/No)": in_preferred,
                "Preferred Unit": preferred_unit,
                "Convertible to SI (Yes/No)": convertible,
                "SI Unit": si_unit,
                "Conversion Formula": conversion_formula
            })

    print(f"✅ Total time taken: {time.time() - start_time:.2f} seconds")
    return results
# Example usage
results = explore_units_for_each_Cohort(
    "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/cohorts"
)
results_df = pd.DataFrame(results)
results_df.to_csv(
    "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/unit_exploration_results.csv",
    index=False
)
print("✅ Results saved to unit_exploration_results.csv")

# merge S7-preferred_units-ABC.csv and lab_units_conversion_standards.csv into one file with preferred units and conversion factors
# import pandas as pd

# def normalize_unit(unit_str):
#     """Normalize UCUM-like units for consistent comparison."""
#     if not isinstance(unit_str, str):
#         return ""
#     unit_str = unit_str.strip().lower()
#     unit_str = (
#         unit_str.replace("ucum:", "")
#         .replace("{counts}", "")
#         .replace("[", "")
#         .replace("]", "")
#         .replace(" ", "")
#     )
#     return unit_str


# # --- File paths ---
# preferred_path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/S7-preferred_units-ABC.csv"
# conversion_path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/lab_units_conversion_standards.csv"
# output_path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/units_conversion.csv"


# # --- Load preferred units ---
# df_pref = pd.read_csv(preferred_path)
# df_pref.columns = [c.lower().strip() for c in df_pref.columns]

# # Normalize OMOP ID
# df_pref["variable omop id"] = (
#     df_pref["variable omop id"]
#     .astype(str)
#     .str.strip()
#     .str.replace(r"\.0$", "", regex=True)
# )


# # --- Load conversion table ---
# df_conv = pd.read_csv(conversion_path)
# df_conv.columns = [c.lower().strip() for c in df_conv.columns]

# # Rename for consistency
# if "analyte" in df_conv.columns:
#     df_conv.rename(columns={"analyte": "variable concept name"}, inplace=True)

# # Normalize OMOP ID and units
# df_conv["variable omop id"] = (
#     df_conv["variable omop id"]
#     .astype(str)
#     .str.strip()
#     .str.replace(r"\.0$", "", regex=True)
# )
# df_conv["conventional units"] = df_conv["conventional units"].astype(str).map(normalize_unit)
# df_conv["si units"] = df_conv["si units"].astype(str).map(normalize_unit)


# # --- Perform full outer merge ---
# merged_df = pd.merge(
#     df_pref,
#     df_conv,
#     on="variable omop id",
#     how="outer",  # ✅ keeps ALL OMOP IDs from both datasets
#     suffixes=("_preferred", "_conversion"),
#     indicator=True
# )

# # ✅ Combine variable concept names intelligently
# if "variable concept name_preferred" in merged_df.columns or "variable concept name_conversion" in merged_df.columns:
#     merged_df["variable concept name"] = merged_df["variable concept name_preferred"].combine_first(
#         merged_df["variable concept name_conversion"]
#     )
#     merged_df.drop(
#         columns=["variable concept name_preferred", "variable concept name_conversion"],
#         inplace=True,
#         errors="ignore"
#     )

# # --- Rename for clarity ---
# merged_df = merged_df.rename(columns={
#     "unit concept code": "preferred_unit_code",
#     "unit concept name": "preferred_unit_name",
#     "conventional to si (multiply by)": "conv_to_si",
#     "si to conventional units(multiply by)": "si_to_conv",
#     "_merge": "merge_origin"
# })

# # --- Reorder columns for readability ---
# cols = [
#     "variable omop id",
#     "variable concept name",
#     "preferred_unit_code",
#     "preferred_unit_name",
#     "conventional units",
#     "conv_to_si",
#     "si units",
#     "si_to_conv",

# ]
# merged_df = merged_df.reindex(columns=cols)

# # --- Save ---
# merged_df.to_csv(output_path, index=False)
# print(f"✅ Merged file saved to: {output_path}")
# print(f"Total OMOP IDs included: {len(merged_df)}")
# print("Merge origin counts:")
# print(merged_df["merge_origin"].value_counts())
