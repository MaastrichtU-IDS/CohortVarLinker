import os
import json
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------
def cell_str(x) -> str:
    """Safe string conversion for pandas cells (handles NaN/None)."""
    return "" if pd.isna(x) else str(x).strip()

def pick_first(*vals) -> str:
    """Return first non-empty string among vals (after cell_str)."""
    for v in vals:
        s = cell_str(v)
        if s:
            return s
    return ""

def to_int_or_none(x):
    """Parse ints robustly (handles NaN, '', 'na', floats-as-strings)."""
    s = cell_str(x)
    if not s:
        return None
    if s.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns and create common aliases so heterogeneous CSVs still work."""
    df.columns = df.columns.str.lower()

    # Common aliases
    if "variablename" not in df.columns:
        if "variable name" in df.columns:
            df["variablename"] = df["variable name"]
        elif "element" in df.columns:
            df["variablename"] = df["element"]
        else:
            df["variablename"] = ""

    if "variablelabel" not in df.columns:
        if "variable label" in df.columns:
            df["variablelabel"] = df["variable label"]
        elif "question_text" in df.columns:
            df["variablelabel"] = df["question_text"]
        elif "question" in df.columns:
            df["variablelabel"] = df["question"]
        elif "query" in df.columns:
            df["variablelabel"] = df["query"]
        elif "element" in df.columns:
            df["variablelabel"] = df["element"]
        else:
            df["variablelabel"] = df["variablename"]

    # If "variable concept name" is missing, but goldname exists (Eye/ADRD/Stroke style)
    if "variable concept name" not in df.columns and "goldname" in df.columns:
        df["variable concept name"] = df["goldname"]

    # Ensure optional columns exist (prevents KeyErrors)
    for col in [
        "domain",
        "units", "unit concept code", "unit concept name", "unit omop id",
        "visits", "visit concept code", "visit concept name", "visit omop id",
        "categorical", "values", "categorical value concept name", "categorical value concept code", "categorical value omop id",
        "additional context concept name", "additional context concept code", "additional context omop id",
        "variable concept code", "variable omop id",
        "goldid"
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    return df


# -----------------------------
# Loaders
# -----------------------------
def load_dictionary(filepath: str) -> pd.DataFrame:
    """Loads a dataset (.sav, .csv, .xlsx) into a DataFrame."""
    if filepath.endswith(".sav"):
        return pd.read_spss(filepath)
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, low_memory=False)
    if filepath.endswith(".xlsx"):
        return pd.read_excel(filepath, sheet_name=0)
    raise ValueError(f"Unsupported file format: {filepath}")

def read_all_excel_files_in_directory(directory_path: str, expected_columns=None) -> pd.DataFrame:
    """Reads all .xlsx and .csv files in a directory and concatenates them."""
    all_data = []

    for file in os.listdir(directory_path):
        if not (file.endswith(".xlsx") or file.endswith(".csv")):
            continue

        file_path = os.path.join(directory_path, file)
        print(f"Reading file: {file_path}")
        df = load_dictionary(file_path)

        if df is None or df.empty:
            continue

        df = normalize_df_columns(df)
        print(df.head())
        all_data.append(df)

    if not all_data:
        print("No Excel/CSV files found.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    if expected_columns:
        combined_df = combined_df.reindex(columns=expected_columns)

    print(f"Total rows combined: {len(combined_df)}")
    return combined_df


# -----------------------------
# Training JSON rows
# -----------------------------
def convert_row_to_json(row: pd.Series) -> dict:
    # Prefer mapped concept name; fallback to goldname; fallback to label/element
    goldid = cell_str(row.get("goldid")).lower()
    mapping_not_found = goldid == "mapping_not_found"

    var_name_dbg = pick_first(row.get("variablename"), row.get("element"))
    input_item = pick_first(
        row.get("variablelabel"),
        row.get("question_text"),
        row.get("element"),
        row.get("variablename"),
        row.get("query"),
    )

    # base_entity: if mapping not found, keep None (but still keep input populated)
    if mapping_not_found:
        base_entity = None
    else:
        base_entity = pick_first(
            row.get("variable concept name"),
            row.get("goldname"),
            row.get("variablelabel"),
            row.get("element"),
        ).lower() or None

    # Additional context entities
    additional_entities = []
    add_ctx = cell_str(row.get("additional context concept name"))
    if add_ctx:
        print(f"Processing additional context: {add_ctx} for variable: {var_name_dbg}")
        if "|" in add_ctx:
            additional_entities.extend([v.strip() for v in add_ctx.lower().split("|") if v.strip()])
        else:
            additional_entities.append(add_ctx.lower())

    # Visit
    visit_label = cell_str(row.get("visit concept name")) or None

    # Categories (raw values) and category labels (mapped names)
    raw_cat = pick_first(row.get("categorical"), row.get("values"))
    categories = []
    if raw_cat:
        if "=" in raw_cat:
            categories = [v.strip().split("=")[-1] for v in raw_cat.lower().split("|") if v.strip()]
        else:
            categories = [v.strip() for v in raw_cat.lower().split("|") if v.strip()]

    raw_cat_lbl = pick_first(row.get("categorical value concept name"))
    categories_label = None
    if raw_cat_lbl:
        categories_label = [v.strip() for v in raw_cat_lbl.lower().split("|") if v.strip()]

    # Units / Visits for input string
    units = cell_str(row.get("units")).lower()
    if raw_cat:
        input_item = f"{input_item}, categorical values: {raw_cat}"
    if units:
        input_item = f"{input_item}, unit: ({units})"
    visits = cell_str(row.get("visits"))
    if visits:
        input_item = f"{input_item}, visit: {visits}"

    domain = cell_str(row.get("domain", "observation")).lower() or None
    print(f"domain: {domain} for variable: {var_name_dbg}")
    
    # If your source row is extremely sparse, at least keep input non-empty
    if not input_item:
        input_item = var_name_dbg or "unknown"
    if base_entity is None:
        return {}
    return {
        "input": input_item,
        "output": json.dumps({
            "domain": domain or "observation",
            "base_entity": base_entity,
            "additional_entities": additional_entities if additional_entities else None,
            "categories": categories_label if categories_label else (categories if categories else None),
            "visit": visit_label,
            "unit": units or None,
            "mapping_status": "not_found" if mapping_not_found else "found"
        })
    }


# -----------------------------
# DB-ready rows
# -----------------------------
LOINC_QUESTIONNAIRE_STR_SEARCH_LIST = [
    "[veterans rand]", "[promis]", "[sf-36]", "[sf-12]",
    "does your health", "are you currently", "do you have any", "are you able to",
    "mos sf-36", "mos sf-12", "mos short form 36", "mos short form 12",
]

def json_data_for_db(df: pd.DataFrame) -> dict:
    result_rows = []
    seen = set()

    for _, row in df.iterrows():
        var_name = pick_first(row.get("variablename"), row.get("element"))
        if not var_name:
            continue

        # Variable label for DB rows (fallback chain)
        var_label_fallback = pick_first(
            row.get("variablelabel"),
            row.get("question_text"),
            row.get("element"),
            row.get("variablename"),
            row.get("query"),
        )

        goldid = cell_str(row.get("goldid")).lower()
        mapping_not_found = goldid == "mapping_not_found"

        # Prefer mapped "variable concept name" (or goldname via normalization),
        # but if mapping not found, still keep a DB row using the human label.
        var_concept_name_raw = cell_str(row.get("variable concept name"))
        if not var_concept_name_raw and not mapping_not_found:
            # If truly missing, fallback to label
            var_concept_name_raw = var_label_fallback

        if mapping_not_found and not var_label_fallback:
            # Nothing useful to store
            continue

        # 1) Variable itself
        if var_concept_name_raw or var_label_fallback:
            variable_concept_name = var_concept_name_raw or var_label_fallback

            # Questionnaire heuristic: use the question/variable label as variable_label
            if any(s in variable_concept_name.lower() for s in LOINC_QUESTIONNAIRE_STR_SEARCH_LIST):
                variable_label = var_label_fallback
            else:
                variable_label = variable_concept_name

            entry = {
                "variable_label": cell_str(variable_label).lower() or None,
                "code": cell_str(row.get("variable concept code")).lower() or None,
                "standard_label": cell_str(var_concept_name_raw).lower() or cell_str(variable_concept_name).lower() or None,
                "omop_id": to_int_or_none(row.get("variable omop id")),
                "mapping_status": "not_found" if mapping_not_found else "found"
            }
            entry_tuple = tuple(entry.items())
            if entry_tuple not in seen:
                seen.add(entry_tuple)
                result_rows.append(entry)

        # 2) Categorical values (only if present)
        cat_raw = pick_first(row.get("categorical"), row.get("values"))
        cat_name_raw = cell_str(row.get("categorical value concept name"))
        if cat_raw and cat_name_raw:
            cat_vals = [v.strip() for v in cat_raw.lower().split("|") if v.strip()]
            cat_codes = [v.strip() for v in cell_str(row.get("categorical value concept code")).lower().split("|")] if cell_str(row.get("categorical value concept code")) else []
            cat_omops = [v.strip() for v in cell_str(row.get("categorical value omop id")).split("|")] if cell_str(row.get("categorical value omop id")) else []
            cat_labels = [v.strip() for v in cat_name_raw.lower().split("|") if v.strip()]

            for categorical_value, concept_code, omop_id, standard_label in zip(cat_vals, cat_codes, cat_omops, cat_labels):
                val = categorical_value.split("=")[-1].strip() if "=" in categorical_value else categorical_value.strip()
                entry = {
                    "variable_label": val,
                    "code": cell_str(concept_code).lower() or None,
                    "standard_label": cell_str(standard_label).lower() or None,
                    "omop_id": to_int_or_none(omop_id),
                    "mapping_status": "found"
                }
                entry_tuple = tuple(entry.items())
                if entry_tuple not in seen:
                    seen.add(entry_tuple)
                    result_rows.append(entry)

        # 3) Additional context
        add_ctx_name = cell_str(row.get("additional context concept name"))
        add_ctx_code = cell_str(row.get("additional context concept code"))
        add_ctx_omop = cell_str(row.get("additional context omop id"))
        if add_ctx_name:
            ctx_names = [v.strip() for v in add_ctx_name.lower().split("|") if v.strip()]
            ctx_codes = [v.strip() for v in add_ctx_code.lower().split("|")] if add_ctx_code else [""] * len(ctx_names)
            ctx_omops = [v.strip() for v in add_ctx_omop.split("|")] if add_ctx_omop else [""] * len(ctx_names)

            for additional_context, concept_code, omop_id in zip(ctx_names, ctx_codes, ctx_omops):
                entry = {
                    "variable_label": additional_context,
                    "code": cell_str(concept_code).lower() or None,
                    "standard_label": additional_context,
                    "omop_id": to_int_or_none(omop_id),
                    "mapping_status": "found"
                }
                entry_tuple = tuple(entry.items())
                if entry_tuple not in seen:
                    seen.add(entry_tuple)
                    result_rows.append(entry)

        # 4) Visit info
        visits_label = cell_str(row.get("visits"))
        visit_code = cell_str(row.get("visit concept code"))
        if visits_label:
            entry = {
                "variable_label": visits_label.lower(),
                "code": visit_code.lower() if visit_code else None,
                "standard_label": cell_str(row.get("visit concept name")).lower() or None,
                "omop_id": to_int_or_none(row.get("visit omop id")),
                "mapping_status": "found"
            }
            entry_tuple = tuple(entry.items())
            if entry_tuple not in seen:
                seen.add(entry_tuple)
                result_rows.append(entry)

        # 5) Unit info
        unit_code = cell_str(row.get("unit concept code"))
        units_label = cell_str(row.get("units"))
        if units_label:
            entry = {
                "variable_label": units_label.lower(),
                "code": unit_code.lower() if unit_code else None,
                "standard_label": cell_str(row.get("unit concept name")).lower() or None,
                "omop_id": to_int_or_none(row.get("unit omop id")),
                "mapping_status": "found"
            }
            entry_tuple = tuple(entry.items())
            if entry_tuple not in seen:
                seen.add(entry_tuple)
                result_rows.append(entry)

    return {"database_data": result_rows}


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    directory = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/alignment_article_cohorts"
    output_file = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/mapping_templates.json"

    df = read_all_excel_files_in_directory(directory)
    if df.empty:
        raise SystemExit("No data loaded. Exiting.")

    df = normalize_df_columns(df)
    print(df.columns.tolist())

    # Training examples
    json_output_all = [convert_row_to_json(row) for _, row in df.iterrows()]
    # remove empty dicts
    json_output_all = [item for item in json_output_all if item]
    # DB rows
    json_output_db = json_data_for_db(df)

    final_json = {
        "all": json_output_all,
        "database_data": json_output_db["database_data"]
    }

    with open(output_file, "w") as f:
        json.dump(final_json, f, indent=4, default=str)

    print(f"JSON output has been saved to {output_file}")