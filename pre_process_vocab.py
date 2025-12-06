
import pandas as pd
from src.utils import load_dictionary
def filter_concepts(concept_csv_file, include_relationshops=['maps to','mapped from','is a','subsumes',
                                                             'atc - snomed eq','rxnorm - snomed eq','icd9p eq - snomed',
                                                            'rxnorm - atc pr lat', 'atc - rxnorm pr lat','has component','component of',
                                                             ]):
    # read  all in lower case
    df = pd.read_csv(concept_csv_file, sep='\t')
    df.columns = df.columns.str.lower()
    print(df.columns)
    df = df.apply(lambda x: x.astype(str).str.lower())
    # print all unique relationships
    print(df['relationship_id'].unique())

    
    df_filtered = df[df['relationship_id'].isin([rel.lower() for rel in include_relationshops])]

    # also filter where concept_id_1 is same as concept_id_2
    df_filtered = df_filtered[df_filtered['concept_id_1'] != df_filtered['concept_id_2']]
    print(f"total concepts: {len(df_filtered)}")
    df_filtered.to_csv("concept_relationship.csv", index=False)

import pandas as pd

def ucum_hierarchy(
    concept_df: pd.DataFrame,
    relationship_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For all UCUM unit concepts, add a parent SNOMED unit concept
    (concept_id = 35624217, concept_code = 767525000) with relationship 'is a'.

    Resulting rows have columns:
      concept_id_1, concept_id_2, relationship_id,
      concept_name_1, concept_name_2,
      valid_start_date, valid_end_date, invalid_reason,
      concept_1_vocabulary, concept_2_vocabulary,
      concept_1_domain, concept_2_domain,
      concept_1_concept_class, concept_2_concept_class,
      concept_code_1, concept_code_2

    If relationship_file is provided, the new rows are appended to it and
    written to output_file.
    """

    # --- Load full CONCEPT table ---
    # concept_df = pd.read_csv(concept_file, sep="\t", dtype=str)
    # concept_df.columns = concept_df.columns.str.lower()

    # Keep useful columns
    concept_df = concept_df[
        [
            "concept_id",
            "concept_name",
            "concept_code",
            "vocabulary_id",
            "domain_id",
            "concept_class_id",
        ]
    ]
    concept_df["concept_id"] = concept_df["concept_id"].astype(str)
    concept_df["vocabulary_id"] = concept_df["vocabulary_id"].str.lower()

    # --- Parent SNOMED unit concept ---
    parent_id = "35624217"         # SNOMED 'unit' concept_id
    parent_row = concept_df[concept_df["concept_id"] == parent_id]
    

    if parent_row.empty:
        raise ValueError(
            f"Parent concept_id {parent_id} not found in data frame. "
            "Check that you have the correct Athena dump."
        )

    parent_row = parent_row.iloc[0]
    parent_name = parent_row["concept_name"]
    parent_code = parent_row["concept_code"]
    parent_vocab = parent_row["vocabulary_id"]
    parent_domain = parent_row["domain_id"]
    parent_class = parent_row["concept_class_id"]

    # --- All UCUM concepts ---
    ucum_df = concept_df[concept_df["vocabulary_id"] == "ucum"].copy()

    if ucum_df.empty:
        raise ValueError("No UCUM concepts found in CONCEPT.csv")

    # --- Build the new relationship rows ---
    # UCUM concepts are the children → concept_id_1
    new_rel = pd.DataFrame(
        {
            "concept_id_1": ucum_df["concept_id"],
            "concept_id_2": parent_id,
            "relationship_id": "is a",  # UCUM unit is-a SNOMED unit
            "concept_name_1": ucum_df["concept_name"],
            "concept_name_2": parent_name,
            "valid_start_date": "1970-01-01",   # arbitrary, but OMOP-ish
            "valid_end_date": "2099-12-31",
            "invalid_reason": "",
            "concept_1_vocabulary": ucum_df["vocabulary_id"],
            "concept_2_vocabulary": parent_vocab,
            "concept_1_domain": ucum_df["domain_id"],
            "concept_2_domain": parent_domain,
            "concept_1_concept_class": ucum_df["concept_class_id"],
            "concept_2_concept_class": parent_class,
            "concept_code_1": ucum_df["concept_code"],
            "concept_code_2": parent_code,
        }
    )

    # --- If an existing relationship file is provided, append to it ---
    if relationship_df is not None:
        existing_rel = relationship_df
        # Make sure columns match / at least share the new_rel columns
        # We align columns by name when concatenating
        combined = pd.concat([existing_rel, new_rel], axis=0, ignore_index=True)
    else:
        combined = new_rel
    return combined
    # combined.to_csv(output_file, index=False)
    # print(
    #     f"Added {len(new_rel)} UCUM→SNOMED 'is a' relationships. "
    #     f"Saved to: {output_file}"
    # )
    
# def add_equivalence_relationships(concept_file, relationship_file, output_file="/Users/komalgilani/Desktop/cmh/data/concept_relationship_equivalence_only.csv"):
#     # if two vocabualaries have same concept_name then we can add equivalence relationship between them e.g. snomed and loinc equivalence will be snomed - loinc eq as well as loinc - snomed eq. Some of these relationships already exist in concept_relationship file but some are missing.


def enrich_relationships(concept_file, relationship_file, icarecvd_vocab_df, output_file="/Users/komalgilani/Desktop/CohortVarLinker/data/concept_relationship.csv"):
    # Load concept and relationship files
    
    include_relationshops = ['is a','subsumes',
                                'rxnorm - atc pr lat', 'atc - rxnorm pr lat',
                                'atc - snomed eq', 'snomed - atc eq',
                                'disposition of' ,'has disposition',
                                "has answer", "answer of", "Component of", "has component", 'cpt4 - snomed eq', 'snomed - cpt4 eq', 'cpt4 - loinc eq' , 'loinc - cpt4 eq'
                                 'rxnorm - atc', 'atc - rxnorm', 'snomed - rxnorm eq','rxnorm - snomed eq'
                                
                                                             ]
    concept_df = pd.read_csv(concept_file, sep='\t', dtype=str)
    concept_df  = concept_df.apply(lambda x: x.astype(str).str.lower())
    print(f"unique invalid reasons {concept_df['invalid_reason'].unique().tolist()}")
    relationship_df = pd.read_csv(relationship_file, sep='\t', dtype=str)
    relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())
    
    concept_df.columns = concept_df.columns.str.lower()
    relationship_df.columns = relationship_df.columns.str.lower()
    print(relationship_df['relationship_id'].unique())
    relationship_df = relationship_df[relationship_df['relationship_id'].isin([rel.lower() for rel in include_relationshops])]
    relationship_df = relationship_df[relationship_df['concept_id_1'] != relationship_df['concept_id_2']]
    
    # concept_df = concept_df[
    # (concept_df['vocabulary_id'].str.lower() == 'snomed') |
    # (concept_df['invalid_reason'] in ['nan', None])
    # ]
    concept_df = concept_df[(concept_df['invalid_reason'] =='nan')]
    print(f"len of total rows {len(concept_df)}")
    concept_df = concept_df[['concept_id', 'concept_name', 'concept_code', 'vocabulary_id', 'domain_id', 'concept_class_id']]
    
    # Merge to get vocabulary and names for concept_id_1
    relationship_df = relationship_df.merge(concept_df, left_on="concept_id_1", right_on="concept_id", how="left")
    relationship_df.rename(columns={"concept_name": "concept_name_1","concept_code": "concept_code_1", "vocabulary_id": "concept_1_vocabulary", "domain_id":"concept_1_domain","concept_class_id":"concept_1_concept_class"}, inplace=True)
    relationship_df.drop(columns=["concept_id"], inplace=True)

    # Merge to get vocabulary and names for concept_id_2
    relationship_df = relationship_df.merge(concept_df, left_on="concept_id_2", right_on="concept_id", how="left")
    relationship_df.rename(columns={"concept_name": "concept_name_2", "concept_code": "concept_code_2", "vocabulary_id": "concept_2_vocabulary","domain_id":"concept_2_domain","concept_class_id":"concept_2_concept_class"}, inplace=True)
    relationship_df.drop(columns=["concept_id"], inplace=True)

    merged_df=pd.concat([relationship_df, icarecvd_vocab_df], axis=0, ignore_index=True)
    # convert all values to lowercase
    # merged_df = merged_df.apply(lambda x: x.astype(str).str.lower())
    merged_df = ucum_hierarchy(concept_df=concept_df, relationship_df=merged_df)
    merged_df.to_csv(output_file, index=False)
    print(f"Enriched file saved as: {output_file}")


def add_parent_child_only(concept_file, relationship_file, output_file="/Users/komalgilani/Desktop/CohortVarLinker/data/concept_relationship_parent_child_only.csv"):
    relationship_df = pd.read_csv(relationship_file, sep=',', dtype=str)
    # filter only 'is a' and 'subsumes' relationships
    concept_df = pd.read_csv(concept_file, sep='\t', dtype=str)
    concept_df  = concept_df.apply(lambda x: x.astype(str).str.lower())
    relationship_df = pd.read_csv(relationship_file, sep='\t', dtype=str)
    relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())
    
    concept_df.columns = concept_df.columns.str.lower()
    relationship_df.columns = relationship_df.columns.str.lower()
    print(relationship_df['relationship_id'].unique())
    relationship_df = relationship_df[relationship_df['relationship_id'].isin([rel.lower() for rel in ['is a','subsumes', 'mapped from', 'maps to']])]
    relationship_df = relationship_df[relationship_df['concept_id_1'] != relationship_df['concept_id_2']]   
    concept_df = concept_df[['concept_id', 'concept_name', 'concept_code', 'vocabulary_id', 'domain_id', 'concept_class_id']]
    
    # Merge to get vocabulary and names for concept_id_1
    relationship_df = relationship_df.merge(concept_df, left_on="concept_id_1", right_on="concept_id", how="left")
    relationship_df.rename(columns={"concept_name": "concept_name_1","concept_code": "concept_code_1", "vocabulary_id": "concept_1_vocabulary", "domain_id":"concept_1_domain","concept_class_id":"concept_1_concept_class"}, inplace=True)
    relationship_df.drop(columns=["concept_id"], inplace=True)

    # Merge to get vocabulary and names for concept_id_2
    relationship_df = relationship_df.merge(concept_df, left_on="concept_id_2", right_on="concept_id", how="left")
    relationship_df.rename(columns={"concept_name": "concept_name_2", "concept_code": "concept_code_2", "vocabulary_id": "concept_2_vocabulary","domain_id":"concept_2_domain","concept_class_id":"concept_2_concept_class"}, inplace=True)
    relationship_df.drop(columns=["concept_id"], inplace=True)
    # relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())
    # Save the enriched file    
    
    icare4cvd_vocb = load_dictionary("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/icare4cvd_vocabulary.xlsx")
 
    merged_df=pd.concat([relationship_df, icare4cvd_vocb], axis=0, ignore_index=True)

    merged_df.to_csv(output_file, index=False)
    print(f"Enriched file saved as: {output_file}")



# def enrich_icare4cvd_vocab(
#     concept_file: str,
#     custom_vocab_file: str,
#     vocab_name: str = "icare4cvd",
# ) -> pd.DataFrame:
#     # Load OMOP concepts
#     concept_df = pd.read_csv(concept_file, sep='\t', dtype=str)
#     concept_df.columns = concept_df.columns.str.lower()
#     concept_df = concept_df[['concept_id', 'concept_code']]
#     concept_df['concept_id'] = concept_df['concept_id'].astype(str)

#     # concept_id -> concept_code lookup
#     id_to_code = dict(zip(concept_df['concept_id'], concept_df['concept_code']))

#     # Load custom vocabulary
#     custom_df = pd.read_excel(custom_vocab_file, dtype=str)
#     custom_df.columns = custom_df.columns.str.lower()
#     # lowercase all string values in custom_df
#     custom_df = custom_df.apply(lambda x: x.astype(str).str.lower())

#     # Make sure id/code/vocab columns exist
#     for col in ['concept_id_1', 'concept_id_2']:
#         if col not in custom_df.columns:
#             custom_df[col] = pd.NA

#     for col in ['concept_code_1', 'concept_code_2']:
#         if col not in custom_df.columns:
#             custom_df[col] = pd.NA

#     for col in ['concept_1_vocabulary', 'concept_2_vocabulary']:
#         if col not in custom_df.columns:
#             custom_df[col] = vocab_name.lower()  # default to custom

#     # Normalize vocab columns to lowercase
#     custom_df['concept_1_vocabulary'] = custom_df['concept_1_vocabulary'].astype(str).str.lower()
#     custom_df['concept_2_vocabulary'] = custom_df['concept_2_vocabulary'].astype(str).str.lower()

#     # --- Fill concept_code_1 only where vocab != icare4cvd ---
#     mask_1 = (
#         custom_df['concept_1_vocabulary'] != vocab_name.lower()
#     ) & custom_df['concept_id_1'].notna() & (
#         custom_df['concept_code_1'].isna()
#         | (custom_df['concept_code_1'].astype(str).str.strip() == '')
#     )

#     custom_df.loc[mask_1, 'concept_code_1'] = (
#         custom_df.loc[mask_1, 'concept_id_1'].astype(str).map(id_to_code)
#     )

#     # --- Fill concept_code_2 only where vocab != icare4cvd ---
#     mask_2 = (
#         custom_df['concept_2_vocabulary'] != vocab_name.lower()
#     ) & custom_df['concept_id_2'].notna() & (
#         custom_df['concept_code_2'].isna()
#         | (custom_df['concept_code_2'].astype(str).str.strip() == '')
#     )

#     custom_df.loc[mask_2, 'concept_code_2'] = (
#         custom_df.loc[mask_2, 'concept_id_2'].astype(str).map(id_to_code)
#     )
    
#     # where concept_1_vocabulary is icare4cvd, create concept_code_1 as 'icare:'+concept_id_1
#     mask_3 = custom_df['concept_1_vocabulary'] == vocab_name.lower()
#     custom_df.loc[mask_3, 'concept_code_1'] = 'icv' + custom_df.loc[mask_3, 'concept_id_1'].astype(str)
#     # where concept_2_vocabulary is icare4cvd, create concept_code_2 as 'icare:'+concept_id_2
#     mask_4 = custom_df['concept_2_vocabulary'] == vocab_name.lower()
#     custom_df.loc[mask_4, 'concept_code_2'] = 'icv' + custom_df.loc[mask_4, 'concept_id_2'].astype(str)      
    
#     return custom_df
       
# def enrich_icare4cvd_vocab(concept_file: str,
#                                   custom_vocab_file: str,
#                                   vocab_name: str = "icare4cvd") -> pd.DataFrame:
#     #  using concept_df we need to add concept_code_1 and concept_code_2 for concept_id_1 and concept_id_2 in custom vocab where concept_1_vocabulary and concept_2_vocabulary != icare4cvd
#     concept_df = pd.read_csv(concept_file, sep='\t', dtype=str)
#     concept_df.columns = concept_df.columns.str.lower()
#     concept_df = concept_df[['concept_id', 'concept_name', 'concept_code',
#                              'vocabulary_id', 'domain_id', 'concept_class_id']]
#     concept_df['concept_id'] = concept_df['concept_id'].astype(str)

#     # Load custom vocabulary
#     custom_df = pd.read_excel(custom_vocab_file, dtype=str)
#     custom_df.columns = custom_df.columns.str.lower()
#     custom_df = custom_df.apply(lambda x: x.astype(str).str.lower())
#     print(custom_df.columns)
    
#     #filter custom_df where concept_1_vocabulary != vocab_name or concept_2_vocabulary != vocab_name
#     custom_df_filtered = custom_df[(custom_df['concept_1_vocabulary'] != vocab_name.lower()) |
#                                    (custom_df['concept_2_vocabulary'] != vocab_name.lower())]
#     print(f"Total rows to enrich: {len(custom_df_filtered)}")
#     # Merge to get concept_code_1 where concept_1_vocabulary != vocab_name
#     custom_df_filtered = custom_df_filtered.merge(
#         concept_df[['concept_id', 'concept_code']],
#         left_on='concept_id_1',
#         right_on='concept_id',
#         how='left',
#         suffixes=('', '_omop1')
#     )
#     custom_df_filtered.rename(columns={"concept_code": "concept_code_1"}, inplace=True)
#     custom_df_filtered.drop(columns=['concept_id'], inplace=True)
#     # Merge to get concept_code_2 where concept_2_vocabulary != vocab_name
#     custom_df_filtered = custom_df_filtered.merge(
#         concept_df[['concept_id', 'concept_code']],
#         left_on='concept_id_2',
#         right_on='concept_id',
#         how='left',
#         suffixes=('', '_omop2')
#     )
#     custom_df_filtered.rename(columns={"concept_code": "concept_code_2"}, inplace=True)
#     custom_df_filtered.drop(columns=['concept_id'], inplace=True)
#     # now merge back with original custom_df to get all rows
#     custom_df_remaining = custom_df[~custom_df.index.isin(custom_df_filtered.index)]
#     custom_df_final = pd.concat([custom_df_filtered, custom_df_remaining], axis=0, ignore_index=True)
#     return custom_df_final
           
# check completences concept_relationship_enriched.csv
# def completences(csv_file):
#     df = pd.read_csv(csv_file, dtype=str)
#     total_rows = len(df)
    
#     # all rows where concept_name_1 exist but concept_code_1 or concept_name_2 or concept_code_2 is missing
#     # missing_concept_1_name = df['concept_name_1'].isna().sum()
#     # missing_concept_2_name = df['concept_name_2'].isna().sum()
#     # missing_concept_1_code = df['concept_code_1'].isna().sum() 
#     # missing_concept_2_code = df['concept_code_2'].isna().sum()
#     missing_df = df[(df['concept_name_1'].notna()) & 
#                     ((df['concept_code_1'].isna()) | (df['concept_name_2'].isna()) | (df['concept_code_2'].isna()))]
#     print(f"Total rows: {total_rows}")
#     missing_df.to_csv("missing_concept_relationships.csv", index=False)
   
def check_concept_code(concept_csv_file):
    df = pd.read_csv(concept_csv_file, sep='\t', dtype=str)
    # df = df.apply(lambda x: x.astype(str).str.lower())
    # check for concept codes that are missing
    df.columns = df.columns.str.lower()
    missing_concept_codes = df[df['concept_code'].isna() | (df['concept_code'].str.strip() == '')]
    
    print(f"Total missing concept codes: {len(missing_concept_codes)}")
    missing_concept_codes.to_csv("missing_concept_codes.csv", index=False)
    

if __name__ == "__main__":
    
    athena_vocab_dir = "/Users/komalgilani/Downloads/athena_vocab_25112025/"
    output_dir = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data"
    
    df = load_dictionary(f"{output_dir}/icare4cvd_vocabulary.csv")
    
    enrich_relationships(
        concept_file=f"{athena_vocab_dir}/CONCEPT.csv",
        relationship_file=f"{athena_vocab_dir}/CONCEPT_RELATIONSHIP.csv",
        icarecvd_vocab_df=df,
        output_file=f"{output_dir}/concept_relationship_enriched.csv",
    )
    