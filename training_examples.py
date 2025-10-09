import pandas as pd
import json
import os

# Read the CSV file
# file_path = "/Users/komalgilani/Desktop/kg_construction/data/MetaData_V2_4_12_2024.csv"  # Replace with the path to your CSV file
# df = pd.read_csv(file_path, delimiter=";")

def load_dictionary( filepath=None) -> pd.DataFrame:
        """Loads the input dataset."""
        if filepath.endswith('.sav'):
            df_input = pd.read_spss(filepath)
            # Optionally save to Excel if needed
         
        elif filepath.endswith('.csv'):
            df_input = pd.read_csv(filepath, low_memory=False)
        elif filepath.endswith('.xlsx'):
            df_input = pd.read_excel(filepath, sheet_name=0)
        else:
            raise ValueError("Unsupported file format.")
        if not df_input.empty:
            return df_input
        else:
            return None

def read_all_excel_files_in_directory(directory_path, expected_columns=None):
    all_data = []
   
    for folder in os.listdir(directory_path):
        # check each sub folder in the directory
        if os.path.isdir(os.path.join(directory_path, folder)):
            sub_directory_path = os.path.join(directory_path, folder)
            print(f"Reading files in subdirectory: {sub_directory_path}")
            for file in os.listdir(sub_directory_path):
                if file.endswith(".xlsx") or file.endswith(".csv"):
                    file_path = os.path.join(sub_directory_path, file)
                    print(f"Reading file: {file_path}")  # Debugging output
                    df = load_dictionary(file_path)
                    df.columns = df.columns.str.lower()  # Convert column names to lowercase
                    print(df.head())  # Print the first few rows of the DataFrame for debugging
                    all_data.append(df)

    if not all_data:
        print("No Excel files found.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # Optional: Reorder or filter columns if expected_columns is provided
    if expected_columns:
        combined_df = combined_df.reindex(columns=expected_columns)

    print(f"Total rows combined: {len(combined_df)}")
    return combined_df

df = read_all_excel_files_in_directory("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/cohorts")  # Replace with the path to your directory
df.columns = df.columns.str.lower()  # Convert column names to lowercase

print(df.columns.tolist())  # Print the first few rows of the DataFrame for debugging
# Function to construct the JSON output

def convert_row_to_json(row):
    base_entity = row["variable concept name"]
    additional_entities = []
    categories = []
    categories_label = None
    visit_label = None
    if not pd.isna(row["additional context concept name"]):
        print(f"Processing additional context: {row['additional context concept name']} for variable: {row['variablename']}")
        if "|" in str(row["additional context concept name"]):
            additional_entities.extend([val.strip() for val in str(row["additional context concept name"]).lower().split("|")])
        else:
            additional_entities.append(str(row["additional context concept name"]).lower())
    if not pd.isna(row["visit concept name"]):
        # additional_entities.append(row["visit concept name"])
        visit_label = row["visit concept name"]

    if not pd.isna(row["categorical"]):
            categories = [val.strip().split("=")[-1] for val in row["categorical"].lower().split("|")]
        
    if not pd.isna(row["categorical value concept name"]):
        categories_label = [val.strip() for val in str(row["categorical value concept name"]).lower().split("|")]
    input_item = row['variablelabel']
    if 'categorical' in row and not pd.isna(row['categorical']):
        input_item = f"{input_item},categorical values: {row['categorical']}"
    if 'units' in row and not pd.isna(row['units']):
        input_item = f"{input_item}, unit: ({row['units'].strip().lower()})"
    if 'visits' in row and not pd.isna(row['visits']):
        input_item = f"{input_item}, visit: {row['visits']}"
    print(f"domain: {row['domain']} for variable: {row['variablename']}")
    result = {
        "input": input_item,
        "output": json.dumps({
            "domain": row["domain"].strip().lower(),
            "base_entity": base_entity,
            "additional_entities": additional_entities if additional_entities else None,
            "categories": categories_label if categories_label else categories if categories else None,
            "visit": visit_label,
            "unit": row["units"].strip().lower() if not pd.isna(row["units"]) else None
        })
    }
    # print(f"Processed row: {result}")  # Debugging output
    return result

# Process the CSV data
json_output = [convert_row_to_json(row) for _, row in df.iterrows()]


# Save the JSON output to a file

LOINC_QUESTIONNAIRE_STR_SEARCH_LIST = [
    
    "[veterans rand]", "[promis]", "[sf-36]", "[sf-12]",
   "does your health", "are you currently", "do you have any","are you able to"
   "mos sf-36", "mos sf-12", "mos short form 36", "mos short form 12",
]
def json_data_for_db(df):
    result_rows = []
    seen = set()
    for _, row in df.iterrows():
        if pd.notna(row.get("variablename")) and row.get("variablename").strip() != "":
            # 1. Variable itself
            if pd.notna(row.get("variable concept name")) and row.get("variable concept name").strip() != "":
                variable_concept_name = row["variable concept name"].strip()
                if any(search_str in variable_concept_name.lower() for search_str in LOINC_QUESTIONNAIRE_STR_SEARCH_LIST):
                    variable_concept_name = row.get("variablelabel", "").strip()
                    entry = {
                        "variable_label": variable_concept_name,
                        "code": row.get("variable concept code", "").strip().lower(),
                        "standard_label": row.get("variable concept name", "").strip().lower(),
                        "omop_id": int(row.get("variable omop id")) if pd.notna(row.get("variable omop id")) and str(row.get("variable omop id")).strip() else None
                    }
                    entry_tuple = tuple(entry.items())
                    if entry_tuple not in seen:
                        seen.add(entry_tuple)
                        result_rows.append(entry)
                print(f"Processing variable: {row['variablename']} with concept name: {row['variable concept code']}")
                entry = {
                        "variable_label": row["variable concept name"].strip().lower(),
                        "code": row.get("variable concept code", "").strip().lower(),
                        "standard_label": row.get("variable concept name", "").strip().lower(),
                        "omop_id": int(row.get("variable omop id")) if pd.notna(row.get("variable omop id")) and str(row.get("variable omop id")).strip() else None
                    }
                entry_tuple = tuple(entry.items())
                if entry_tuple not in seen:
                    seen.add(entry_tuple)
                    result_rows.append(entry)

            # 2. Each categorical value (if present)
            if pd.notna(row.get("categorical")) and pd.notna(row.get("categorical value concept code")):
                print(f"Processing categorical values for variable: {row['categorical']} for variable: {row['variablename']}")
                for categorical_value, concept_code, omop_id, standard_label in zip(
                    row.get("categorical", "").lower().split("|"),
                    row.get("categorical value concept code", "").lower().split("|"),
                    str(row.get("categorical value omop id", "")).split("|"),
                    row.get("categorical value concept name", "").lower().split("|")
                ):
                    entry = {
                        "variable_label": categorical_value.strip().split("=")[-1] if "=" in categorical_value else categorical_value.strip(),
                        "code": concept_code.strip(),
                        "standard_label": standard_label.strip() if standard_label else None,
                        "omop_id": int(omop_id) if pd.notna(omop_id) and str(omop_id).strip() and omop_id != "na" else None
                    }
                    entry_tuple = tuple(entry.items())
                    if entry_tuple not in seen:
                        seen.add(entry_tuple)
                        result_rows.append(entry)
                        
            # 3. Additional context (if present)
            if pd.notna(row.get("additional context concept code")) and pd.notna(row.get("additional context concept name")) and pd.notna(row.get("additional context omop id")):
                
                print(f"Processing additional context: {row['additional context omop id']} for variable: {row['variablename']}")
                for additional_context, concept_code, omop_id, standard_label in zip(
                    str(row.get("additional context concept name", "")).lower().split("|"),
                    row.get("additional context concept code", "").lower().split("|"),
                    str(row.get("additional context omop id", "")).split("|"),
                    str(row.get("additional context concept name", "")).lower().split("|")
                ):
                   
                    entry = {
                        "variable_label": additional_context.strip().lower(),
                        "code": concept_code.strip().lower(),
                        "standard_label": standard_label.strip().lower() if standard_label else None,
                        "omop_id": int(float(omop_id)) if pd.notna(omop_id) and str(omop_id).strip() else None
                    }
                    entry_tuple = tuple(entry.items())
                    if entry_tuple not in seen:
                        seen.add(entry_tuple)
                        result_rows.append(entry)

            # 3. Visit info
            if pd.notna(row.get("visits")) and pd.notna(row.get("visit concept code")):
                entry = {
                    "variable_label": row.get("visits", "").strip().lower(),
                    "code": row.get("visit concept code", "").strip().lower(),
                    "standard_label": row.get("visit concept name", "").strip().lower(),
                    "omop_id": int(row.get("visit omop id")) if pd.notna(row.get("visit omop id")) and str(row.get("visit omop id")) else None
                }
                entry_tuple = tuple(entry.items())
                if entry_tuple not in seen:
                    seen.add(entry_tuple)
                    result_rows.append(entry)

            # 4. Unit info
            if pd.notna(row.get("unit concept code")) and pd.notna(row.get("units")):
                entry = {
                    "variable_label": row.get("units", "").strip().lower(),
                    "code": row.get("unit concept code", "").strip().lower(),
                    "standard_label": row.get("unit concept name", "").strip().lower(),
                    "omop_id": int(row.get("unit omop id")) if pd.notna(row.get("unit omop id")) and str(row.get("unit omop id")) else None
                }
                entry_tuple = tuple(entry.items())
                if entry_tuple not in seen:
                    seen.add(entry_tuple)
                    result_rows.append(entry)

    return {
        "database_data": result_rows
    }

    
json_output1 = json_data_for_db(df)
output_file = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/mapping_templates.json"
# add both json_output and json_output1 to the same file
json_output = {
    "all": json_output,  # original training examples
    "database_data": json_output1["database_data"]  # newly created DB-ready rows
}

with open(output_file, 'w') as f:
    json.dump(json_output, f, indent=4, default=str)
print(f"JSON output has been saved to {output_file}")