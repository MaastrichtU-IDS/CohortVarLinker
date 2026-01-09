
import pandas as pd
from experiment.utils import load_dictionary
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


# def add_snomed_atc_equivalence_viarxnorm(
#     concept_file,
#     concept_syn_file,
#     relationship_file,
#     output_file="/Users/komalgilani/Desktop/CohortVarLinker/data/concept_relationship_snomed_atc_equivalence_only.csv",
# ):
#     """
#     Creates SNOMED<->ATC equivalence edges using this exact rule:

#     SNOMED_substance_parent  (parent of)  SNOMED_child  --[SNOMED-RxNorm eq]-->  RxNorm
#     RxNorm  --[RxNorm-ATC pr lat]-->  ATC_leaf  (child of)  ATC_parent

#     Output edges:
#       SNOMED_substance_parent --[snomed - atc eq]--> ATC_parent
#       ATC_parent              --[atc - snomed eq]--> SNOMED_substance_parent

#     Plus: connect SNOMED disposition of the substance to the same ATC_parent:
#       SNOMED_disposition --[snomed - atc eq]--> ATC_parent
#       ATC_parent         --[atc - snomed eq]--> SNOMED_disposition

#     IMPORTANT:
#       - We DO NOT connect to ATC_leaf.
#       - We roll each ATC_leaf up by 1 hop to its direct parent (via subsumes).
#       - If RxNorm maps to multiple ATC_leaf, we prefer the COMMON direct parent (intersection).
#     """

#     # -----------------------------
#     # Load + normalize
#     # -----------------------------
#     concept_df = pd.read_csv(concept_file, sep="\t", dtype=str)
#     relationship_df = pd.read_csv(relationship_file, sep="\t", dtype=str)

#     concept_df.columns = concept_df.columns.str.lower()
#     relationship_df.columns = relationship_df.columns.str.lower()

#     concept_df = concept_df.apply(lambda x: x.astype(str).str.lower())
#     relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())

#     concept_df = enrich_concept_synonyms(concept_df, concept_syn_file)

#     # -----------------------------
#     # Helpers
#     # -----------------------------
#     def build_child_to_parents(df_subsumes):
#         """Given edges parent --subsumes--> child, return dict child -> set(parents)."""
#         m = {}
#         for p, c in zip(df_subsumes["concept_id_1"].tolist(), df_subsumes["concept_id_2"].tolist()):
#             m.setdefault(c, set()).add(p)
#         return m

#     def pick_preferred_snomed_parents(child_id, parents_set, concept_meta):
#         """
#         Prefer 'drug substance' parents if available; otherwise return all parents.
#         Fallback to {child_id} if no parents.
#         """
#         if not parents_set:
#             return {child_id}

#         drug_substance = set()
#         for pid in parents_set:
#             row = concept_meta.get(pid)
#             if not row:
#                 continue
#             cname = row.get("concept_name", "")
#             cclass = row.get("concept_class_id", "")
#             # heuristic preference
#             if ("drug substance" in cclass) or ("(drug substance)" in cname):
#                 drug_substance.add(pid)

#         return drug_substance if drug_substance else set(parents_set)

#     def rxnorm_to_common_atc_parent(rxnorm_atc_leaves_df, atc_child_to_parents):
#         """
#         For each RxNorm, map to ATC direct parents.
#         If RxNorm has multiple ATC leaves:
#           - if intersection of parent-sets is non-empty -> keep intersection (common parent(s))
#           - else -> keep union
#         Returns df: [rxnorm_id, atc_parent_id]
#         """
#         rows = []
#         grouped = rxnorm_atc_leaves_df.groupby("rxnorm_id")["atc_leaf_id"].apply(list).to_dict()

#         for rxn, leaves in grouped.items():
#             parent_sets = []
#             for leaf in leaves:
#                 ps = atc_child_to_parents.get(leaf, set())
#                 # if no parent, treat leaf as its own "parent" (or you can skip)
#                 parent_sets.append(ps if ps else {leaf})

#             common = set.intersection(*parent_sets) if parent_sets else set()
#             chosen = common if common else set.union(*parent_sets) if parent_sets else set()

#             for p in chosen:
#                 rows.append((rxn, p))

#         return pd.DataFrame(rows, columns=["rxnorm_id", "atc_parent_id"]).drop_duplicates()

#     # -----------------------------
#     # Concept meta lookup
#     # -----------------------------
#     concept_meta_cols = [
#         "concept_id", "concept_name", "vocabulary_id", "domain_id",
#         "concept_class_id", "concept_code", "synonyms"
#     ]
#     concept_meta_df = concept_df[concept_meta_cols].drop_duplicates()
#     concept_meta = concept_meta_df.set_index("concept_id").to_dict(orient="index")

#     # -----------------------------
#     # Relationship slices
#     # -----------------------------
#     rel = relationship_df

#     # SNOMED child -> RxNorm
#     snomed_rxnorm = rel[rel["relationship_id"].isin(["snomed - rxnorm eq"])][
#         ["concept_id_1", "concept_id_2"]
#     ].drop_duplicates().rename(columns={"concept_id_1": "snomed_child_id", "concept_id_2": "rxnorm_id"})

#     # RxNorm -> ATC leaf
#     rxnorm_atc = rel[rel["relationship_id"].isin(["rxnorm - atc pr lat"])][
#         ["concept_id_1", "concept_id_2"]
#     ].drop_duplicates().rename(columns={"concept_id_1": "rxnorm_id", "concept_id_2": "atc_leaf_id"})

#     # Hierarchy edges: parent --subsumes--> child
#     subsumes = rel[rel["relationship_id"].str.contains("subsumes", na=False)][
#         ["concept_id_1", "concept_id_2"]
#     ].drop_duplicates()

#     # Normalize disposition edges to: substance_snomed_id -> disposition_snomed_id
#     disp_has = rel[rel["relationship_id"].isin(["has disposition"])][
#         ["concept_id_1", "concept_id_2"]
#     ].rename(columns={"concept_id_1": "substance_snomed_id", "concept_id_2": "disposition_snomed_id"})

#     disp_of = rel[rel["relationship_id"].isin(["disposition of", "is disposition of"])][
#         ["concept_id_1", "concept_id_2"]
#     ].rename(columns={"concept_id_1": "disposition_snomed_id", "concept_id_2": "substance_snomed_id"})

#     disposition_edges = pd.concat([disp_has, disp_of], ignore_index=True).drop_duplicates()

#     # -----------------------------
#     # Build SNOMED child -> SNOMED parent (1 hop)
#     # -----------------------------
#     snomed_ids = set(concept_meta_df.loc[concept_meta_df["vocabulary_id"] == "snomed", "concept_id"].tolist())
#     atc_ids = set(concept_meta_df.loc[concept_meta_df["vocabulary_id"] == "atc", "concept_id"].tolist())

#     snomed_subsumes = subsumes[
#         subsumes["concept_id_1"].isin(snomed_ids) &
#         subsumes["concept_id_2"].isin(snomed_ids)
#     ].drop_duplicates()

#     atc_subsumes = subsumes[
#         subsumes["concept_id_1"].isin(atc_ids) &
#         subsumes["concept_id_2"].isin(atc_ids)
#     ].drop_duplicates()

#     snomed_child_to_parents = build_child_to_parents(snomed_subsumes)
#     atc_child_to_parents = build_child_to_parents(atc_subsumes)

#     # Expand snomed_rxnorm to include preferred substance parent(s) of the SNOMED child
#     # Result: snomed_substance_id -> rxnorm_id
#     snomed_parent_rows = []
#     for child_id, rxn_id in zip(snomed_rxnorm["snomed_child_id"], snomed_rxnorm["rxnorm_id"]):
#         parents = snomed_child_to_parents.get(child_id, set())
#         chosen_parents = pick_preferred_snomed_parents(child_id, parents, concept_meta)
#         for p in chosen_parents:
#             snomed_parent_rows.append((p, rxn_id))

#     snomed_substance_rxnorm = pd.DataFrame(
#         snomed_parent_rows, columns=["snomed_substance_id", "rxnorm_id"]
#     ).drop_duplicates()

#     # -----------------------------
#     # Replace ATC leaf with ATC direct parent (1 hop), prefer common parent if multiple leaves
#     # -----------------------------
#     # keep only ATC leaves that truly belong to ATC vocabulary (safety)
#     rxnorm_atc = rxnorm_atc[rxnorm_atc["atc_leaf_id"].isin(atc_ids)].drop_duplicates()

#     rxnorm_to_atc_parent = rxnorm_to_common_atc_parent(rxnorm_atc, atc_child_to_parents)

#     # -----------------------------
#     # Build SNOMED substance -> ATC parent (via RxNorm)
#     # -----------------------------
#     substance_to_atc_parent = snomed_substance_rxnorm.merge(
#         rxnorm_to_atc_parent, on="rxnorm_id", how="inner"
#     )[["snomed_substance_id", "atc_parent_id"]].drop_duplicates()

#     # -----------------------------
#     # Add disposition -> ATC parent (disposition of the substance)
#     # -----------------------------
#     disp_to_atc_parent = disposition_edges.merge(
#         substance_to_atc_parent,
#         left_on="substance_snomed_id",
#         right_on="snomed_substance_id",
#         how="inner"
#     )[["disposition_snomed_id", "atc_parent_id"]].drop_duplicates()

#     # -----------------------------
#     # Combine (substance + disposition) as SNOMED->ATC-parent
#     # -----------------------------
#     snomed_to_atc_parent = pd.concat([
#         substance_to_atc_parent.rename(columns={"snomed_substance_id": "concept_id_1", "atc_parent_id": "concept_id_2"}),
#         disp_to_atc_parent.rename(columns={"disposition_snomed_id": "concept_id_1", "atc_parent_id": "concept_id_2"})
#     ], ignore_index=True).drop_duplicates()

#     # -----------------------------
#     # Create bidirectional equivalence relationships
#     # -----------------------------
#     rel_fwd = snomed_to_atc_parent.copy()
#     rel_fwd["relationship_id"] = "snomed - atc eq"

#     rel_rev = snomed_to_atc_parent.rename(columns={"concept_id_1": "concept_id_2", "concept_id_2": "concept_id_1"}).copy()
#     rel_rev["relationship_id"] = "atc - snomed eq"

#     new_relationships_final = pd.concat([rel_fwd, rel_rev], ignore_index=True).drop_duplicates()

#     # -----------------------------
#     # Attach metadata (names/vocab/etc.) like your original function
#     # -----------------------------
#     concept_meta_small = concept_meta_df.copy()

#     out = new_relationships_final.merge(
#         concept_meta_small,
#         left_on="concept_id_1",
#         right_on="concept_id",
#         how="left"
#     ).rename(columns={
#         "concept_name": "concept_name_1",
#         "vocabulary_id": "concept_1_vocabulary",
#         "domain_id": "concept_1_domain",
#         "concept_class_id": "concept_1_concept_class",
#         "concept_code": "concept_code_1",
#         "synonyms": "concept_synonym_1",
#     }).drop(columns=["concept_id"])

#     out = out.merge(
#         concept_meta_small,
#         left_on="concept_id_2",
#         right_on="concept_id",
#         how="left"
#     ).rename(columns={
#         "concept_name": "concept_name_2",
#         "vocabulary_id": "concept_2_vocabulary",
#         "domain_id": "concept_2_domain",
#         "concept_class_id": "concept_2_concept_class",
#         "concept_code": "concept_code_2",
#         "synonyms": "concept_synonym_2",
#     }).drop(columns=["concept_id"])

#     out.to_csv(output_file, index=False)
#     print(f"[OK] Saved SNOMED<->ATC (parent-rolled) equivalences to: {output_file}")
#     print(f"[INFO] Rows written: {len(out):,}")


def add_snomed_atc_equivalence_viarxnorm(concept_file, concept_syn_file, relationship_file, output_file="/Users/komalgilani/Desktop/CohortVarLinker/data/concept_relationship_snomed_atc_equivalence_only.csv"):
    # use snomed to rxnom relationship and rxnorm to atc relationship to create snomed to atc equivalence relationship
    concept_df = pd.read_csv(concept_file, sep='\t', dtype=str)
    concept_df  = concept_df.apply(lambda x: x.astype(str).str.lower())
    relationship_df = pd.read_csv(relationship_file, sep='\t', dtype=str)
    relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())    
    concept_df.columns = concept_df.columns.str.lower()
    concept_df = enrich_concept_synonyms(concept_df, concept_syn_file)
    relationship_df.columns = relationship_df.columns.str.lower()
    print(relationship_df['relationship_id'].unique())
    # get snomed to rxnorm relationships
    snomed_rxnorm_df = relationship_df[relationship_df['relationship_id'].isin(['snomed - rxnorm eq'])]
    # get rxnorm to atc relationships
    rxnorm_atc_df = relationship_df[relationship_df['relationship_id'].isin(['rxnorm - atc pr lat'])]
    
     # all snomed concept that are rxnorm equivalent and have relationship 'is disposition of' with other snomed concepts should be add as equivalence relationship with atc concepts which are lat pr of those rxnorm concepts.

    # merge on concept_id_2 of snomed_rxnorm and concept_id_1 of rxnorm_atc
    
    merged_df = snomed_rxnorm_df.merge(rxnorm_atc_df, left_on="concept_id_2", right_on="concept_id_1", how="inner", suffixes=('_snomed_rxnorm', '_rxnorm_atc'))
    # create new dataframe with columns
   
    
    new_relationships = pd.DataFrame()
    new_relationships['concept_id_1'] = merged_df['concept_id_1_snomed_rxnorm']
    new_relationships['concept_id_2'] = merged_df['concept_id_2_rxnorm_atc']
    # make bidirectional relationship snomed - atc eq and atc - snomed eq
    new_relationships_1 = new_relationships.copy()
    new_relationships_1['relationship_id'] = 'snomed - atc eq'
    new_relationships_2 = new_relationships.copy()
    new_relationships_2['relationship_id'] = 'atc - snomed eq'
    new_relationships_final = pd.concat([new_relationships_1, new_relationships_2], axis=0, ignore_index=True)
    # merge to get concept names and vocabularies
    concept_df = concept_df[['concept_id', 'concept_name', 'vocabulary_id', 'domain_id', 'concept_class_id', 'concept_code','synonyms']]
    # Merge to get vocabulary and names for concept_id_1
    new_relationships_final = new_relationships_final.merge(concept_df, left_on="concept_id_1", right_on="concept_id", how="left")
    new_relationships_final.rename(columns={"concept_name": "concept_name_1", "vocabulary_id": "concept_1_vocabulary","domain_id":"concept_1_domain","concept_class_id":"concept_1_concept_class","concept_code":"concept_code_1", 'synonyms': 'concept_synonym_1'}, inplace=True)
    new_relationships_final.drop(columns=["concept_id"], inplace=True)  
    # Merge to get vocabulary and names for concept_id_2
    new_relationships_final = new_relationships_final.merge(concept_df, left_on="concept_id_2", right_on="concept_id", how="left")
    new_relationships_final.rename(columns={"concept_name": "concept_name_2", "vocabulary_id": "concept_2_vocabulary","domain_id":"concept_2_domain","concept_class_id":"concept_2_concept_class","concept_code":"concept_code_2", "synonyms": "concept_synonym_2"}, inplace=True)
    new_relationships_final.drop(columns=["concept_id"], inplace=True)  
    new_relationships_final.to_csv(output_file, index=False)
    print(f"Snomed-ATC equivalence relationships saved as: {output_file}")
    
def enrich_concept_synonyms(concept_df, concept_syn_file): # Changed first arg to concept_df
    # We assume concept_df is already a DataFrame
    
    # Load synonym file
    concept_syn = pd.read_csv(concept_syn_file, sep='\t', dtype=str)
    concept_syn = concept_syn.apply(lambda x: x.astype(str).str.lower())
    concept_syn.columns = concept_syn.columns.str.lower()
    
    # Merge synonyms: filter for English (4180186)
    # Note: Ensure concept_df columns are lowercased before this function is called or do it here
    merged_df = concept_df.merge(
        concept_syn[concept_syn['language_concept_id'] == '4180186'], 
        left_on="concept_id", 
        right_on="concept_id", 
        how="left", 
        suffixes=('', '_syn')
    )
    
    # Group by concept_id to aggregate synonyms
    def aggregate_synonyms(syn_series):
        syn_list = syn_series.dropna().unique().tolist()
        return ";".join(syn_list) if syn_list else ""
    
    # Define columns to keep (ensure these exist in your concept_df)
    agg_dict = {
        'concept_name': 'first',
        'concept_code': 'first',
        'vocabulary_id': 'first',
        'domain_id': 'first',
        'concept_class_id': 'first',
        'concept_synonym_name': aggregate_synonyms
    }
    
    aggregated_df = merged_df.groupby('concept_id').agg(agg_dict).reset_index()
    
    # Rename synonym column
    aggregated_df.rename(columns={'concept_synonym_name': 'synonyms'}, inplace=True)
    
    # RETURN the dataframe so the caller can use it
    return aggregated_df

def add_eq_for_share_synonyms(concept_file, concept_syn_file, relationship_file, output_file="/Users/komalgilani/Desktop/CohortVarLinker/data/concept_relationship_equivalence_synonyms_only.csv"):
    concept_df = pd.read_csv(concept_file, sep='\t', dtype=str)
    concept_df  = concept_df.apply(lambda x: x.astype(str).str.lower())
    relationship_df = pd.read_csv(relationship_file, sep='\t', dtype=str)
    relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())    
    concept_df.columns = concept_df.columns.str.lower()
    concept_df = enrich_concept_synonyms(concept_df, concept_syn_file)
    # create a dictionary of synonym to list of concept_ids
    syn_dict = {}
    for index, row in concept_df.iterrows():
        syns = row['synonyms'].split(';')
        for syn in syns:
            if syn not in syn_dict:
                syn = syn.strip().lower()
                syn_dict[syn] = set()
            syn_dict[syn].add(row['concept_id'])
    # create equivalence relationships for concepts sharing synonyms
    new_relationships = []
    for syn, concept_ids in syn_dict.items():
        concept_ids = list(concept_ids)
        if len(concept_ids) > 1:
            for i in range(len(concept_ids)):
                for j in range(i+1, len(concept_ids)):
                    new_relationships.append({
                        'concept_id_1': concept_ids[i],
                        'concept_id_2': concept_ids[j],
                        'relationship_id': 'maps to'
                    })
                    new_relationships.append({
                        'concept_id_1': concept_ids[j],
                        'concept_id_2': concept_ids[i],
                        'relationship_id': 'mapped from'
                    })
    new_relationships_df = pd.DataFrame(new_relationships)
    # merge to get concept names and vocabularies
    concept_df = concept_df[['concept_id', 'concept_name', 'vocabulary_id', 'domain_id', 'concept_class_id', 'concept_code','synonyms']]
    # Merge to get vocabulary and names for concept_id_1
    new_relationships_df = new_relationships_df.merge(concept_df, left_on="concept_id_1", right_on="concept_id", how="left")
    new_relationships_df.rename(columns={"concept_name": "concept_name_1", "vocabulary_id": "concept_1_vocabulary","domain_id":"concept_1_domain","concept_class_id":"concept_1_concept_class","concept_code":"concept_code_1", 'synonyms': 'concept_synonym_1'}, inplace=True)
    new_relationships_df.drop(columns=["concept_id"], inplace=True)  
    # Merge to get vocabulary and names for concept_id_2
    new_relationships_df = new_relationships_df.merge(concept_df, left_on="concept_id_2", right_on="concept_id", how="left")
    new_relationships_df.rename(columns={"concept_name": "concept_name_2", "vocabulary_id": "concept_2_vocabulary","domain_id":"concept_2_domain","concept_class_id":"concept_2_concept_class","concept_code":"concept_code_2", "synonyms": "concept_synonym_2"}, inplace=True)
    new_relationships_df.drop(columns=["concept_id"], inplace=True)  
    new_relationships_df.to_csv(output_file, index=False)
    print(f"Equivalence relationships based on shared synonyms saved as: {output_file}")
def enrich_relationships(concept_file, concept_syn_file, relationship_file, icarecvd_vocab_df, output_file="/Users/komalgilani/Desktop/CohortVarLinker/data/concept_relationship.csv"):
    # Load concept and relationship files
    
    include_relationshops = ['is a','subsumes',
                                'rxnorm - atc pr lat', 'atc - rxnorm pr lat',
                                'rxnorm - atc','atc - rxnorm',
                                'atc - snomed eq', 'snomed - atc eq',
                                'disposition of' ,'has disposition',
                                "has answer", "answer of", "Component of", "has component", 
                                'cpt4 - snomed eq', 'snomed - cpt4 eq', 
                                'cpt4 - loinc eq' , 'loinc - cpt4 eq',
                                'snomed - rxnorm eq','rxnorm - snomed eq'
                                
                                                             ]
    concept_df = pd.read_csv(concept_file, sep='\t', dtype=str)
    # concept_syn = pd.read_csv(concept_syn_file, sep='\t', dtype=str)
    concept_df  = concept_df.apply(lambda x: x.astype(str).str.lower())
    print(f"unique invalid reasons {concept_df['invalid_reason'].unique().tolist()}")
    relationship_df = pd.read_csv(relationship_file, sep='\t', dtype=str)
    relationship_df = relationship_df.apply(lambda x: x.astype(str).str.lower())
    
    concept_df.columns = concept_df.columns.str.lower()
    relationship_df.columns = relationship_df.columns.str.lower()
    print(relationship_df['relationship_id'].unique())
    relationship_df = relationship_df[relationship_df['relationship_id'].isin([rel.lower() for rel in include_relationshops])]
    relationship_df = relationship_df[relationship_df['concept_id_1'] != relationship_df['concept_id_2']]
    concept_df = concept_df[(concept_df['invalid_reason'] =='nan')]
    print(f"len of total rows {len(concept_df)}")
    concept_df = concept_df[['concept_id', 'concept_name', 'concept_code', 'vocabulary_id', 'domain_id', 'concept_class_id']]
    concept_df = enrich_concept_synonyms(concept_df, concept_syn_file)
    # Merge to get vocabulary and names for concept_id_1
    relationship_df = relationship_df.merge(concept_df, left_on="concept_id_1", right_on="concept_id", how="left")
    relationship_df.rename(columns={"concept_name": "concept_name_1","concept_code": "concept_code_1", "vocabulary_id": "concept_1_vocabulary", "domain_id":"concept_1_domain","concept_class_id":"concept_1_concept_class", "synonyms": "concept_synonym_1"}, inplace=True)
    relationship_df.drop(columns=["concept_id"], inplace=True)

    # Merge to get vocabulary and names for concept_id_2
    relationship_df = relationship_df.merge(concept_df, left_on="concept_id_2", right_on="concept_id", how="left")
    relationship_df.rename(columns={"concept_name": "concept_name_2", "concept_code": "concept_code_2", "vocabulary_id": "concept_2_vocabulary","domain_id":"concept_2_domain","concept_class_id":"concept_2_concept_class", "synonyms": "concept_synonym_2"}, inplace=True)
    relationship_df.drop(columns=["concept_id"], inplace=True)

    # concept df where the concepts dont have any relationships like they are standalone concepts
    standalone_concepts = concept_df[~concept_df['concept_id'].isin(relationship_df['concept_id_1'].tolist() + relationship_df['concept_id_2'].tolist())]
    # create their relationship toselves
    standalone_concepts_self_rel = standalone_concepts.copy()
    standalone_concepts_self_rel['concept_id_1'] = standalone_concepts_self_rel['concept_id']
    standalone_concepts_self_rel['concept_id_2'] = standalone_concepts_self_rel['concept_id']
    standalone_concepts_self_rel['relationship_id'] = 'maps to'
    standalone_concepts_self_rel.rename(columns={"concept_name": "concept_name_1","concept_code": "concept_code_1", "vocabulary_id": "concept_1_vocabulary", "domain_id":"concept_1_domain","concept_class_id":"concept_1_concept_class", "synonyms": "concept_synonym_1"}, inplace=True)
    standalone_concepts_self_rel.rename(columns={"concept_name": "concept_name_2","concept_code": "concept_code_2", "vocabulary_id": "concept_2_vocabulary", "domain_id":"concept_2_domain","concept_class_id":"concept_2_concept_class", "synonyms": "concept_synonym_2"}, inplace=True)
    # standalone_concepts_self_rel.drop(columns=["concept_id"], inplace=True)
    relationship_df = pd.concat([relationship_df, standalone_concepts_self_rel], axis=0, ignore_index=True)
    # print(f"Total standalone concepts: {len(standalone_concepts)}")
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
    # icare4cvd_vocb.rename(columns={"concept_synonym_name": "concept_synonym_1"}, inplace=True)
    merged_df=pd.concat([relationship_df, icare4cvd_vocb], axis=0, ignore_index=True)

    merged_df.to_csv(output_file, index=False)
    print(f"Enriched file saved as: {output_file}")



def check_concept_code(concept_csv_file):
    df = pd.read_csv(concept_csv_file, sep='\t', dtype=str)
    # df = df.apply(lambda x: x.astype(str).str.lower())
    # check for concept codes that are missing
    df.columns = df.columns.str.lower()
    missing_concept_codes = df[df['concept_code'].isna() | (df['concept_code'].str.strip() == '')]
    
    print(f"Total missing concept codes: {len(missing_concept_codes)}")
    missing_concept_codes.to_csv("missing_concept_codes.csv", index=False)
    
def explore_relationships(relation_df:pd.DataFrame, concept_id:int):
    # explore all relationships for a given concept_id
    concept_id_str = str(concept_id)
    rel_1 = relation_df[relation_df['concept_id_1'] == concept_id_str]
    rel_2 = relation_df[relation_df['concept_id_2'] == concept_id_str]
    print(f"Total relationships where concept_id_1 == {concept_id}: {len(rel_1)}")
    # only get rows where 'eq' in relationship_id
    rel_1 = rel_1[rel_1['relationship_id'].str.contains('eq')]
    print(rel_1)
    rel_2 = rel_2[rel_2['relationship_id'].str.contains('eq')]
    
    print(f"Total relationships where concept_id_2 == {concept_id}: {len(rel_2)}")
    print(rel_2)
if __name__ == "__main__":
    
    athena_vocab_dir = "/Users/komalgilani/Downloads/athena_vocab_25112025/"
    output_dir = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data"
    add_snomed_atc_equivalence_viarxnorm(
        concept_file=f"{athena_vocab_dir}/CONCEPT.csv",
        concept_syn_file=f"{athena_vocab_dir}/CONCEPT_SYNONYM.csv",
        relationship_file=f"{athena_vocab_dir}/CONCEPT_RELATIONSHIP.csv",
        output_file=f"{output_dir}/concept_relationship_snomed_atc_equivalence_only.csv",
    )
    # add_eq_for_share_synonyms(  
    #                           concept_file=f"{athena_vocab_dir}/CONCEPT.csv",
    #                           concept_syn_file=f"{athena_vocab_dir}/CONCEPT_SYNONYM.csv",
    #                           relationship_file=f"{athena_vocab_dir}/CONCEPT_RELATIONSHIP.csv",
    #                           output_file=f"{output_dir}/concept_relationship_equivalence_synonyms_only.csv",
    #                           )
    
    df = load_dictionary(f"{output_dir}/icare4cvd_vocabulary.csv")
    
    enrich_relationships(
        concept_file=f"{athena_vocab_dir}/CONCEPT.csv",
        concept_syn_file=f"{athena_vocab_dir}/CONCEPT_SYNONYM.csv",
        relationship_file=f"{athena_vocab_dir}/CONCEPT_RELATIONSHIP.csv",
        icarecvd_vocab_df=df,
        output_file=f"{output_dir}/concept_relationship_enriched.csv",
    )
    df = pd.read_csv("/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/concept_relationship_enriched.csv", dtype=str)
    explore_relationships(df, 1112807)
    
    snomed_atc_df = pd.read_csv(f"{output_dir}/concept_relationship_snomed_atc_equivalence_only.csv", dtype=str)
    enrich_df = pd.read_csv(f"{output_dir}/concept_relationship_enriched.csv", dtype=str)
    # eq_df = pd.read_csv(f"{output_dir}/concept_relationship_equivalence_synonyms_only.csv", dtype=str)
    # check how many rows are common between snomed_atc_df and enrich_df
    # combine all rows
    combined_df = pd.concat([snomed_atc_df, enrich_df], axis=0, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.to_csv(f"{output_dir}/concept_relationship_enriched.csv", index=False)
    print(f"Final enriched file saved as: {output_dir}/concept_relationship_enriched.csv")
    print(f"Total rows in final enriched file: {len(combined_df)}")
    