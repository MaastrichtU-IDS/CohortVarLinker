# # #!/usr/bin/env python3
# # """
# # Simple UCUM unit conversion using NIH UCUM REST API.
# # Example:
# #     python ucum_converter.py 90 mg/dL mmol/L
# # """

# # import sys
# # import requests

# # def ucum_convert(value, from_unit, to_unit):
# #     """
# #     Convert a numeric value from one UCUM unit to another via NIH UCUM API.
# #     If molecular weight or charge are required (mass↔mol↔eq) and not provided,
# #     UCUM may return an error.
# #     """
# #     base_url = "https://ucum.nlm.nih.gov/ucum-service/v1/ucumtransform"
# #     url = f"{base_url}/{value}/from/{from_unit}/to/{to_unit}"
# #     headers = {"Accept": "application/json"}

# #     try:
# #         response = requests.get(url, headers=headers, timeout=10)
# #         if response.ok:
# #             data = response.json()
# #             result = data["UCUMWebServiceResponse"]["Response"]["ResultQuantity"]
# #             print(f"\n✅ {value} {from_unit} = {result} {to_unit}\n")
# #             return float(result)
# #         else:
# #             print(f"❌ UCUM Error {response.status_code}: {response.text}")
# #     except Exception as e:
# #         print(f"⚠️ UCUM conversion failed: {e}")

# # if __name__ == "__main__":
# #     value = 35
# #     from_unit = "mg/dL"
# #     to_unit = "mmol/L"

# #     ucum_convert(value, from_unit, to_unit)



# import xml.etree.ElementTree as ET

# def load_ucum_units(xml_path):
#     """
#     Parse ucum-essence.xml and return a dictionary of UCUM units and conversion factors.
#     """
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
    
#     ucum_dict = {}
#     for unit in root.findall(".//unit"):
#         code = unit.get("Code")
#         value = unit.get("Value")
#         base_unit = unit.get("Unit")
#         property = unit.findtext("property")
#         symbol = unit.findtext("printSymbol")
        
#         if code and value:
#             ucum_dict[code] = {
#                 "factor": float(value),
#                 "base_unit": base_unit,
#                 "property": property,
#                 "symbol": symbol
#             }
#     return ucum_dict

# def convert_ucum_local(value, from_unit, to_unit, ucum_dict):
#     if from_unit not in ucum_dict or to_unit not in ucum_dict:
#         print(f"⚠️ Unknown UCUM code(s): {from_unit}, {to_unit}")
#         return None

#     from_info = ucum_dict[from_unit]
#     to_info = ucum_dict[to_unit]
    
#     # Ensure they share the same property (e.g., pressure, mass, etc.)
#     if from_info["property"] != to_info["property"]:
#         print(f"❌ Cannot convert between different properties: {from_info['property']} vs {to_info['property']}")
#         return None

#     base_value = float(value) * from_info["factor"]     # convert to base units
#     result = base_value / to_info["factor"]             # convert to target
#     print(f"✅ {value} {from_unit} = {result} {to_unit}")
#     return result

# ucum_dict = load_ucum_units("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/ucum-essence.xml")

# print("mm[Hg]:", ucum_dict["mm[Hg]"])
# print("g:", ucum_dict["g"])
# print("kPa:", ucum_dict["kPa"])


# convert_ucum_local(35, "mg/dL", "mmol/L", ucum_dict)


# import pyreadr

# path = "/Users/komalgilani/lab2clean/data/loinc_reference_unit_v1.rda"
# result = pyreadr.read_r(path)

# df = list(result.values())[0]  # extract the DataFrame
# print(df.head())
# print(df.columns)
# df.to_csv("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/loinc_reference_unit_v1.csv", index=False)


# import pandas as pd

# import pandas as pd
# import math

# class UnitConverter:
#     def __init__(self, csv_path):
#         self.df = pd.read_csv(csv_path)
#         # normalize UCUM codes (upper-case, no extra spaces)
#         self.df['csCode_'] = self.df['csCode_'].astype(str).str.strip()
#         self.df['magnitude_'] = pd.to_numeric(self.df['magnitude_'], errors='coerce')
        
#         # build lookup
#         self.lookup = {
#             row['csCode_']: row['magnitude_']
#             for _, row in self.df.iterrows()
#             if pd.notna(row['magnitude_'])
#         }
#         print(f"✅ Loaded {len(self.lookup)} UCUM units with conversion factors.")

#     def convert(self, value, from_unit, to_unit):
#         from_unit = from_unit.strip()
#         to_unit = to_unit.strip()

#         if from_unit not in self.lookup or to_unit not in self.lookup:
#             print(f"⚠️ Unit not found: {from_unit} or {to_unit}")
#             return None

#         from_mag = self.lookup[from_unit]
#         to_mag = self.lookup[to_unit]

#         # Compute factor relative to SI base
#         factor = from_mag / to_mag
#         return value * factor


# # Example usage
# converter = UnitConverter("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/parsed_units_df.csv")
# result = converter.convert(35, "mg/dL", "mmol/L")
# print(f"Conversion result: {result}")   

# ucum_df = pd.read_csv("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/parsed_units_df.csv")

# ucum_dict = {}
# for _, row in ucum_df.iterrows():
#     unit = str(row.get("ucum_code")).strip()
#     magnitude = row.get("magnitude_") or row.get("conversion_factor") or 1
#     if pd.notna(unit):
#         ucum_dict[unit] = float(magnitude)
# print(f"Loaded {len(ucum_dict)} UCUM units with conversion factors.")

# def convert_unit(value, from_unit, to_unit):
#     """
#     Convert numeric value between UCUM-compatible units
#     using the magnitude ratios from parsed_units_df.csv
#     """
#     from_unit = from_unit.strip()
#     to_unit = to_unit.strip()

#     if from_unit == to_unit:
#         print(f"✅ {from_unit} and {to_unit} are identical.")
#         return value

#     if from_unit not in ucum_dict or to_unit not in ucum_dict:
#         print(f"⚠️ Unit(s) not found: {from_unit}, {to_unit}")
#         return None

#     from_mag = ucum_dict[from_unit]
#     to_mag = ucum_dict[to_unit]

#     factor = from_mag / to_mag
#     result = value * factor

#     print(f"✅ {value} {from_unit} = {result} {to_unit} (factor = {factor})")
#     return result

# # Example usage
# result=convert_unit(35, "mg/dL", "mmol/L")
# print("Conversion result:", result)



import pandas as pd

def load_preferred_units(csv_path):

    # Preferred unit definitions per OMOP concept
    preferred_df = pd.read_csv(csv_path)
    preferred_df.columns = [c.strip().lower() for c in preferred_df.columns]
    preferred_df["concept_name.x.x"] = preferred_df["concept_name.x.x"].str.strip().str.lower()
    preferred_df["concept_name.y"] = preferred_df["concept_name.y"].str.strip().str.lower()
    preferred_df["concept_code.x"] = preferred_df["concept_code.x"].astype(str)

    # Keep relevant columns
    preferred_units = preferred_df[[
        "concept_id",          # OMOP ID
        "concept_code.x",      # LOINC code
        "concept_name.x.x",    # analyte name
        "unit_concept_id",     # OMOP unit concept id
        "concept_name.y",      # unit name
        "vocabulary_id.x"
    ]].rename(columns={
        "concept_name.x.x": "analyte",
        "concept_name.y": "preferred_unit_name",
        "concept_id": "omop_id",
        "concept_code.x": "loinc_code"
    })
    preferred_units["analyte"] = preferred_units["analyte"].str.strip().str.lower()
    preferred_units.to_csv("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/preferred_units.csv", index=False)
    print(f"✅ Loaded {len(preferred_units)} preferred units from {csv_path}")
    return preferred_units


def load_conversion_factors(csv_path):
    labcorp_df = pd.read_csv(csv_path)
    labcorp_df.columns = [c.strip().lower().replace(" ", "_") for c in labcorp_df.columns]
    labcorp_df["analyte"] = labcorp_df["analyte"].str.strip().str.lower()

    conversion_lookup = {}
    for _, row in labcorp_df.iterrows():
        if pd.notna(row["conventional_to_si_(multiply_by)"]):
            try:
                factor = float(str(row["conventional_to_si_(multiply_by)"]).replace(",", "."))
            except:
                factor = None
        else:
            factor = None
        conversion_lookup[row["analyte"]] = {
            "from": str(row["conventional_units"]).strip().lower(),
            "to": str(row["si_units"]).strip().lower(),
            "factor": factor
        }

    print(f"✅ Loaded {len(conversion_lookup)} LabCorp conversion entries")


def convert_to_preferred_unit(omop_id, analyte, value, current_unit, preferred_units, conversion_lookup):
    """
    Convert a numeric lab result into its preferred unit.
    Uses preferred_unit.csv for target unit and LabCorp conversion table for factors.
    """
    analyte = analyte.lower().strip()
    current_unit = current_unit.lower().strip()

    
    # --- 1️⃣ Find preferred unit from preferred-unit table
    row = preferred_units.loc[preferred_units["omop_id"] == omop_id]
    if row.empty:
        print(f"⚠️ No preferred unit found for OMOP {omop_id}")
        return value, current_unit

    preferred_unit = row["preferred_unit_name"].values[0].lower().strip()

    # --- 2️⃣ If already in preferred unit
    if current_unit == preferred_unit:
        return value, preferred_unit

    # --- 3️⃣ Look for analyte-specific conversion in LabCorp table
    if analyte in conversion_lookup:
        rule = conversion_lookup[analyte]
        if (rule["from"] == current_unit and rule["to"] == preferred_unit and rule["factor"]):
            new_val = value * rule["factor"]
            return new_val, preferred_unit
        elif (rule["to"] == current_unit and rule["from"] == preferred_unit and rule["factor"]):
            new_val = value / rule["factor"]
            return new_val, preferred_unit

    print(f"⚠️ No conversion rule found for {analyte} ({current_unit} → {preferred_unit})")
    return value, preferred_unit

prefered_units_df = load_preferred_units("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/preferred_unit.csv")
labcorp_conversion_lookup = load_conversion_factors("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/labcorp_unit_conversion.csv")
result, new_unit = convert_to_preferred_unit(
    omop_id=3013762,
    analyte="Glucose",
    value=90,
    current_unit="mg/dL",
    preferred_units=prefered_units_df,
    conversion_lookup=labcorp_conversion_lookup
)