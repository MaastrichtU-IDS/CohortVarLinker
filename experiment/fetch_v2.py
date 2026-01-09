from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
# from SPARQLWrapper import JSON, SPARQLWrapper
from functools import lru_cache
from collections import defaultdict
from .config import settings
from .utils import apply_rules, execute_query
from .vector_db import search_in_db
from .constraints import MappingType, EmbeddingType 
import json
from .query_builder import SPARQLQueryBuilder
 
DATE_HINTS = ["visit date", "date of visit","date of event"]
SIMILARITY_THRESHOLD = 0.8
# we may later seperate "6 months prior to baseline", "prior to baseline visit" as a match but not to baseline time
DERIVED_VARIABLES_LIST= [
    
     {
                    "name": "BMI-derived",
                    "omop_id": 3038553,           
                    "code": "loinc:39156-5",
                    "label": "Body mass index (BMI) [Ratio]",
                    "unit": "kg/m2",
                    "required_omops": [3016723, 3025315],
                    "category": "measurement",
                    "data_type": "continuous_variable"
                },
                {
                    "name": "eGFR_CG-derived",
                    "omop_id": 37169169,          
                    "code": "snomed:1556501000000100",
                    "label": "Estimated creatinine clearance calculated using actual body weight Cockcroft-Gault formula",
                    "unit": "ml/min",
                    "required_omops": [3016723, 3022304, 46235213],
                    "category": "measurement",
                    "data_type": "continuous_variable"
                }
]
@lru_cache(maxsize=None)
def check_visit_string(visit_str_src: str, visit_str_tgt:str) -> str:
    # if src or tgt visit string contains any of the time hints, return the value of the visit that is not in time hint
    # print(f"Checking visit strings: src='{visit_str_src}', tgt='{visit_str_tgt}'")
    
    for hint in DATE_HINTS:
        if hint in visit_str_src.lower():
            return visit_str_tgt
        if hint in visit_str_tgt.lower():
            return visit_str_src

    # for hint in BASELINE_TIME_HINTS_V1:
    #     if hint in visit_str_src.lower() or hint in visit_str_tgt.lower():
    #         if 'follow-up' in visit_str_src.lower() or 'follow-up' in visit_str_tgt.lower():
    #             return visit_str_src
    #         return 'baseline time'
    return visit_str_src


def _parse_bindings(bindings: Iterable[Dict[str, Any]]) -> tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
]:
    """Return source elements, target elements and exact matches."""
    source_elems, target_elems, matches = [], [], []
    for result in bindings:
        src_category = result["source_domain"]["value"].strip().lower()
        tgt_category = result["target_domain"]["value"].strip().lower()
        omop = int(result["omop_id"]["value"])
        code_label = result["code_label"]["value"]
        code_value = result["code_value"]["value"]
        src_raw_list = result["source_data"]["value"].split(" | ") if result["source_data"]["value"] else []
        tgt_raw_list = result["target_data"]["value"].split(" | ") if result["target_data"]["value"] else []
        def split_var_visit(raw_str):
            # Check if "[visit]" pattern exists at the end
            if "[" in raw_str and raw_str.endswith("]"):
                parts = raw_str.rsplit("[", 1) # Split from right to ensure variable names with '[' don't break it
                var_name = parts[0].strip()
                visit_val = parts[1].strip(" ]")
                return var_name, visit_val
            return raw_str.strip(), "baseline time" # Default visit if missing

        # 3. Apply helper
        src_vars, src_visits = [], []
        if len(src_raw_list) > 0:
            for s in src_raw_list:
                v, vis = split_var_visit(s)
                src_vars.append(v)
                src_visits.append(vis)

        tgt_vars, tgt_visits = [], []
        if len(tgt_raw_list) > 0:
            for t in tgt_raw_list:
                v, vis = split_var_visit(t)
                tgt_vars.append(v)
                tgt_visits.append(vis)
        assert len(src_vars) == len(src_visits), (
            f"Visit column Length mismatch with variable labels: {len(src_vars)} != {len(src_visits)}"
        )
        # print(f"tgt vars: {tgt_vars}, tgt visits: {tgt_visits}")
        assert len(tgt_vars) == len(tgt_visits), (
            f"Visit column Length mismatch with variable labels: {len(tgt_vars)} != {len(tgt_visits)}"
        )
        matches.extend(
            _exact_match_records(src_vars, tgt_vars, src_visits, tgt_visits,
                                omop, code_value, code_label, src_category, tgt_category)
        )
        source_elems.extend(
            _build_elements("source", src_vars, src_visits, omop, code_value, code_label, src_category)
        )
        target_elems.extend(
            _build_elements("target", tgt_vars, tgt_visits, omop, code_value, code_label, category=tgt_category
                            )
        )
    print(f"Parsed {len(source_elems)} source elements, {len(target_elems)} target elements, and {len(matches)} exact matches." )
    return source_elems, target_elems, matches


def _build_elements(
    role: str,
    variables: List[str],
    visits: List[str],
    omop_id: int,
    code: str,
    code_label: str,
    category: str,
) -> List[dict[str, Any]]:
    return [
        {
            'omop_id': omop_id,
            'code': code.strip(),
            'code_label': code_label,
            role: el,
            'category': category,
            'visit': vis,
        }
        for el, vis in zip(variables, visits)
    ]

def get_var_context(code_label_str) -> str:
    """Combine composite OMOP fields into a single pipe-separated string.
    Format: omop_id|code_label|code_value
    """
    # print(f"Getting var context for code_label_str: {code_label_str}")
    if not code_label_str or code_label_str.strip() == "":
        return None
    else: 
        parts = [part.strip() for part in code_label_str.split("|")][1:] #skip first part 
        combined = "|".join(parts)
        return combined


def _exact_match_records(
    src_vars: List[str],
    tgt_vars: List[str],
    src_visits: List[str],
    tgt_visits: List[str],
    omop: int,
    code_value: str,
    code_label: str,
    src_category: str,
    tgt_category: str,
) -> List[Dict[str, Any]]:
    """Return records where source and target share OMOP id and visit."""
    res = []
    for s, sv in zip(src_vars, src_visits):
        for t, tv in zip(tgt_vars, tgt_visits):
            # sv = check_visit_string(sv)
            # tv = check_visit_string(tv)
            if check_visit_string(sv, tv) == check_visit_string(tv, sv):
                category = src_category if src_category == tgt_category else f"{src_category}|{tgt_category}"
                res.append(
                    {
                        "source": s,
                        "target": t,
                        "somop_id": omop,
                        "tomop_id": omop,
                        "scode": code_value.strip(),
                        "slabel": code_label,
                        "tcode": code_value.strip(),
                        "tlabel": code_label,
                        "category": category,
                        "mapping_relation": "skos:exactMatch",
                        "source_visit": sv,
                        "target_visit": tv,
                    }
                )
    # print(res)
    return res


def extend_with_derived_variables(single_source: dict, 
                                  standard_derived_variable: tuple, 
                                  parameters_omop_ids: list, 
                                  variable_name: str, category: str) -> dict:
    """
    If source and target both can derive a standard variable (e.g. BMI, eGFR),
    create a new mapping row for the derived variable.
    `single_source` should have keys 'source', 'target', and 'mapped' (all lists of dicts).
    `standard_derived_variable`: tuple(code, label, omop_id)
    `parameters_omop_ids`: list of OMOP IDs needed to compute variable.
    `variable_name`: string, e.g., "bmi"
    """
    single_source = single_source.copy()  # Avoid in-place mutation

    def find_omop_id_rows(data: list, omop_code: str, code_key: str = "omop_id") -> list:
        found = []
        for row in data:
            code_value = row.get(code_key, "")
            if int(code_value) == int(omop_code):
                found.append(row)
        return found

    def can_produce_variable(data: dict, parameters_codes: list, side: str = "source") -> bool:
        code_key = "somop_id" if side == "source" else "tomop_id"
        has_parameter_un_mapped = all(
            len(find_omop_id_rows(data[side], code, code_key="omop_id")) > 0 for code in parameters_codes
        )
        has_parameters_mapped = all(
            len(find_omop_id_rows(data['mapped'], code, code_key=code_key)) > 0 for code in parameters_codes
        )
        return has_parameter_un_mapped or has_parameters_mapped

    source_derived_rows = find_omop_id_rows(single_source["source"], standard_derived_variable[2], code_key="omop_id")
    target_derived_rows = find_omop_id_rows(single_source["target"], standard_derived_variable[2], code_key="omop_id")

    # If neither side has the variable, and cannot produce it, do nothing
    if not source_derived_rows and not target_derived_rows:
        return {}
    source_can = can_produce_variable(single_source, parameters_omop_ids, side="source")
    target_can = can_produce_variable(single_source, parameters_omop_ids, side="target")
    if not (source_can and target_can):
        return {}

    if source_derived_rows:
        source_varname = source_derived_rows[0]["source"]
    else:
        source_varname = f"{variable_name}(derived)"
    if target_derived_rows:
        target_varname = target_derived_rows[0]["target"]
    else:
        target_varname = f"{variable_name} (derived)"

    mapping_type = "skos:relatedMatch" if ("derived" in source_varname.lower() or "derived" in target_varname.lower()) else "skos:exactMatch"
    return {
        "source": source_varname,
        "target": target_varname,
        "somop_id": standard_derived_variable[2],
        "tomop_id": standard_derived_variable[2],
        "scode": standard_derived_variable[0],
        "slabel": standard_derived_variable[1],
        "tcode": standard_derived_variable[0],
        "tlabel": standard_derived_variable[1],
        "mapping_relation": mapping_type,
        "source_visit": "baseline time",
        "target_visit": "baseline time",
        "category": category,
        "transformation_rule": {
            "description": f"Derived variable {variable_name} using variable columns  {parameters_omop_ids} from original dataset. Consider the timeline of the longitudinal data when using this variable.",
        }
        # "harmonization_status": "Complete Match (Compatible)"
    }

def _neuro_symbolic_matcher(
    src: List[dict[str, Any]],
    tgt: List[dict[str, Any]],
    graph: Any,
    vector_db: Any,
    embed_model: Any,
    target_study: str,
    collection_name: str,
) -> List[Dict[str, Any]]:
    """Match remaining variables using the OMOP graph or embedding search."""
    final: List[Dict[str, Any]] = []
    
    # Pre-index Source and Target for faster lookup
    # Note: We index by ID only, not (ID, category), to allow cross-category matching
    src_map: Dict[int, List[dict[str, Any]]] = defaultdict(list)
    tgt_map: Dict[int, List[dict[str, Any]]] = defaultdict(list)

    for el in src:
        src_map[el["omop_id"]].append(el)
    for el in tgt:
        tgt_map[el["omop_id"]].append(el)

    unique_targets: set[int] = {el["omop_id"] for el in tgt}

    # Iterate over unique Source IDs
    for sid, s_elems in src_map.items():
        # Identify potential targets (all targets except the source itself)
        tgt_ids = unique_targets - {sid}
        if not tgt_ids:
            continue
            
        # Use the first label for searching (assuming homogeneity)
        label = s_elems[0]["code_label"]
        # Use the first category for domain context in search (optional, can be relaxed)
        # If you want to search ACROSS domains in vector DB, remove 'data_domain' filter below
        category = s_elems[0]["category"] 
        
        reachable = None
        # Graph Search: Check paths to any target ID
        reachable = graph.source_to_targets_paths(sid, tgt_ids, max_depth=1)
       
        if reachable:
            matched = reachable
        else:
            # Vector Search: Fallback if no graph path
            # We still use the source category to narrow the search space initially, 
            # but you can pass None to 'data_domain' to search everything.
            matched_db = set(
                search_in_db(
                    vectordb=vector_db,
                    embedding_model=embed_model,
                    query_text=label,
                    target_study=[target_study],
                    limit=100,
                    data_domain=[category], # Keep this if you want to find "Drugs" for "Drugs"
                    min_score=SIMILARITY_THRESHOLD,
                    collection_name=collection_name,
                )
            )
            matched = set()
            for tid in matched_db:
                matched.add((tid, "skos:relatedMatch"))

        # Construct Match Records
        for tid, mapping_relation in matched:
            if tid not in tgt_map:
                continue
                
            for se in s_elems:
                for te in tgt_map[tid]:
                    tv = te['visit']
                    sv = se['visit']
                    
                    # --- CHANGE: REMOVED EXACT CATEGORY MATCH CONSTRAINT ---
                    # Old: if se["category"] != te["category"] or ...
                    
                    # Only check Visit constraint now
                    if check_visit_string(sv, tv) != check_visit_string(tv, sv):
                        continue
                        
                    # Construct composite category string if they differ
                    src_cat = se.get("category", "")
                    tgt_cat = te.get("category", "")
                    combined_category = src_cat if src_cat == tgt_cat else f"{src_cat}|{tgt_cat}"

                    final.append(
                        {
                            "source": se.get("source", ""),
                            "target": te.get("target", ""),
                            "source_visit": sv,
                            "target_visit": tv,
                            "somop_id": se.get("omop_id", ""),
                            "tomop_id": te.get("omop_id", ""),
                            "scode": se.get("code", ""),
                            "slabel": se.get("code_label", ""),
                            "tcode": te.get("code", ""),
                            "tlabel": te.get("code_label", ""),
                            "category": combined_category, # Store the potentially different categories
                            "mapping_relation": mapping_relation
                        }
                    )
    return final


# def _graph_vector_matches(
#     src: List[dict[str, Any]],
#     tgt: List[dict[str, Any]],
#     vector_db: Any,
#     embed_model: Any,
#     target_study: str,
#     collection_name: str,
#     hierarchy_map: Dict[int, Dict[int, str]] # <--- NEW ARGUMENT
# ) -> List[Dict[str, Any]]:
#     """
#     Match variables using 1) Pre-fetched Hierarchy Graph, then 2) Vector Embeddings.
#     """
#     final: List[Dict[str, Any]] = []
    
#     # Index Targets by ID and Category for fast lookup
#     tgt_map: Dict[tuple, List[dict[str, Any]]] = defaultdict(list)
#     for el in tgt:
#         tgt_map[(el["omop_id"], el["category"])].append(el)

#     # Process Sources
#     for se in src:
#         sid = se["omop_id"]
#         category = se["category"]
#         label = se["code_label"]
        
#         found_matches = set()
        
#         # --- STRATEGY 1: CHECK HIERARCHY GRAPH ---
#         # Look at the pre-fetched related IDs for this Source ID
#         related_concepts = hierarchy_map.get(sid, {})
        
#         for tid, relation in related_concepts.items():
#             # Check if this related ID exists in our Target variables
#             target_key = (tid, category)
#             if target_key in tgt_map:
#                 found_matches.add((tid, relation))

#         # --- STRATEGY 2: VECTOR SEARCH (Fallback) ---
#         # If no hierarchy match found, OR if we want to augment with fuzzy matches
#         if not found_matches:
#             matched_db = set(
#                 search_in_db(
#                     vectordb=vector_db,
#                     embedding_model=embed_model,
#                     query_text=label,
#                     target_study=[target_study],
#                     limit=100,
#                     data_domain=[category],
#                     min_score=SIMILARITY_THRESHOLD,
#                     collection_name=collection_name,
#                 )
#             )
#             for tid in matched_db:
#                 # Vector search implies "Related" or "Close" match usually
#                 found_matches.add((tid, "skos:relatedMatch"))

#         # --- BUILD RESULT ROWS ---
#         for tid, mapping_relation in found_matches:
#             target_key = (tid, category)
#             if target_key not in tgt_map:
#                 continue
                
#             for te in tgt_map[target_key]:
#                 tv = te['visit']
#                 sv = se['visit']
                
#                 # Check Visit Constraints
#                 if check_visit_string(sv, tv) != check_visit_string(tv, sv):
#                     continue
                    
#                 final.append({
#                     "source": se.get("source", ""),
#                     "target": te.get("target", ""),
#                     "source_visit": sv,
#                     "target_visit": tv,
#                     "somop_id": sid,
#                     "tomop_id": tid,
#                     "scode": se.get("code", ""),
#                     "slabel": se.get("code_label", ""),
#                     "tcode": te.get("code", ""),
#                     "tlabel": te.get("code_label", ""),
#                     "category": category,
#                     "mapping_relation": mapping_relation
#                 })
                
#     return final

def fetch_variables_attributes(var_names_list: list[str], study_name: str, graph_repo: str) -> pd.DataFrame:
    """
    Fetches statistical metadata for variables in parallel to optimize network waiting time.
    """
    data_dict = []
    
    # create chunks of 50 variables
    var_names_list_chunks = [var_names_list[i:i + 50] for i in range(0, len(var_names_list), 50)]

    def process_chunk(var_list):
        """Helper function to process a single chunk of variables."""
        chunk_results = []
        try:
            values_str = " ".join(f'"{v}"' for v in var_list)
            # query = _build_statistic_query(study_name, values_str, graph_repo)
            query = SPARQLQueryBuilder.build_statistic_query(study_name, values_str, graph_repo)
            # print(query)
            # Execute query (this is the blocking network call)
            # print(f"Executing statistic query for chunk of size {len(var_list)}") 
            results = execute_query(query)
            
            bindings = results.get("results", {}).get("bindings", [])
            for result in bindings:
                identifier = result['identifier']['value']
            
                if identifier in var_list: # Ensure strictly matching current chunk context
                    composite_value = get_var_context(result['code_label']['value']) if 'code_label' in result else None
                    
                    chunk_results.append({
                        'identifier': identifier,
                        'stat_label': result['stat_label']['value'] if 'stat_label' in result and result['stat_label']['value'].strip() != "" else None,
                        'unit_label': result['unit_label']['value'] if 'unit_label' in result  and result['unit_label']['value'].strip() != "" else None,
                        'data_type': result['data_type_val']['value'] if 'data_type_val' in result  and result['data_type_val']['value'].strip() != "" else None,
                        "categories_labels": result['all_cat_labels']['value'] if 'all_cat_labels' in result  and result['all_cat_labels']['value'].strip() != "" else None,
                        'original_categories': result['all_original_cat_values']['value'] if 'all_original_cat_values' in result and result['all_original_cat_values']['value'].strip() != "" else None,
                        'composite':  composite_value
                    })
        except Exception as e:
            print(f"Error fetching statistics for chunk: {e}")
        
        return chunk_results

    # Use ThreadPoolExecutor to run chunks in parallel
    # max_workers=5 is usually a safe balance between speed and not overwhelming the endpoint
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all chunks to the pool
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in var_names_list_chunks}
        
        # Collect results as they finish
        for future in as_completed(future_to_chunk):
            try:
                result_data = future.result()
                data_dict.extend(result_data)
            except Exception as exc:
                print(f"Chunk execution generated an exception: {exc}")

    data_dict = pd.DataFrame.from_dict(data_dict)
    print(f"Fetched statistics for {len(data_dict)} variables.")
    # print(f"head of data dict: {data_dict.head()}")
    return data_dict     

def _check_variable_attributes(
    df: pd.DataFrame, source_vars: List[str], target_vars: List[str], src_study: str, tgt_study: str, graph_repo: str
) -> pd.DataFrame:
    src_stats = fetch_variables_attributes(source_vars, src_study, graph_repo)
    tgt_stats = fetch_variables_attributes(target_vars, tgt_study, graph_repo)


    if not src_stats.empty and "identifier" in src_stats.columns:
        df = df.merge(
            src_stats.rename(
                columns={
                    "identifier": "source",
                    "stat_label": "source_type",
                    "unit_label": "source_unit",
                    "data_type": "source_data_type",
                    "categories_labels": "source_categories_labels",
                    "original_categories": "source_original_categories",
                    "composite": "source_composite",
                }
            ),
            on="source",
            how="left",
        )
    else:
        print(f"Warning: No source statistics found for {src_study}")
    if not tgt_stats.empty and "identifier" in tgt_stats.columns:
        df = df.merge(
            tgt_stats.rename(
                columns={
                    "identifier": "target",
                    "stat_label": "target_type",
                    "unit_label": "target_unit",
                    "data_type": "target_data_type",
                    "categories_labels": "target_categories_labels",
                    "original_categories": "target_original_categories",
                    "composite": "target_composite"
                }
            ),
            on="target",
            how="left",
        )
    else:
        print(f"Warning: No target statistics found for {tgt_study}")
    return df
def map_source_target(
    source_study_name: str,
    target_study_name: str,
    vector_db: Any,
    embedding_model: Any,
    graph_db_repo: str = "https://w3id.org/CMEO/graph",
    collection_name: str = "studies_metadata",
    graph: Any = None,
    mapping_mode: str = MappingType.OO.value,
) -> pd.DataFrame:
    """
    Align variables between two studies using  bidirectional graph created using OMOP controlled vocabularies, 
    semantic embedding similarity and rules-based approaches. 
    """
   
    default_columns = [
            "source", "target", "somop_id", "tomop_id", "scode", "slabel", "tcode", "tlabel", "category", "source_visit", "target_visit", "source_type", "source_unit", "source_data_type", "source_categories_codes", "source_original_categories", "target_type", "target_unit", "target_data_type", "target_categories_codes", "target_original_categories", "mapping_relation", "transformation_rule", "harmonization_status"
        ]
    # query = _build_alignment_query(source_study_name, target_study_name, graph_db_repo)
    print(f"Building mappings between {source_study_name} and {target_study_name} using mode {mapping_mode}...")
    query = SPARQLQueryBuilder.build_alignment_query(source_study_name, target_study_name, graph_db_repo)
    bindings = execute_query(query)

    source_elems, target_elems, matches = _parse_bindings(bindings['results']['bindings'])
    # print(f"Source elements: {len(source_elems)}, Target elements: {len(target_elems)}, Matches: {len(matches)}")
    if not target_elems and not matches:
        # print(f"No matches found for {source_study_name} and {target_study_name}.")
        columns = [
            f"{source_study_name}_variable",
            f"{target_study_name}_variable",
            "somop_id",
            "tomop_id",
            "scode",
            "slabel",
            "tcode",
            "tlabel",
            "category",
            "mapping_relation",
            "source_visit",
            "target_visit",
        ]
        return pd.DataFrame(columns=columns)

    
    
    
    # Build up your matching dict
    single_source = {
        "source": source_elems,
        "target": target_elems,
        "mapped": matches
    }
    
    if mapping_mode in [MappingType.OEH.value, MappingType.OED.value, MappingType.OEC.value]:
        matches.extend(_neuro_symbolic_matcher(
            source_elems,
            target_elems,
            graph,
            vector_db,
            embedding_model,
            target_study_name,
            collection_name,
        ))
    for derived in DERIVED_VARIABLES_LIST:
        derived_row = extend_with_derived_variables(
            single_source=single_source,
            standard_derived_variable=(derived["code"], derived["label"], derived["omop_id"]),
            parameters_omop_ids=derived["required_omops"],
            variable_name=derived["name"],
            category=derived["category"],
        )
        if derived_row:
            matches.append(derived_row)
    print(f"Total matches found: {len(matches)}")
    
    df = pd.DataFrame(matches).drop_duplicates(subset=["source", "target"])
    if df.empty:
        # print(f"No matches found for {source_study_name} and {target_study_name}.")
        df = pd.DataFrame(columns=default_columns)
        return df
    df = _check_variable_attributes(
        df,
        df["source"].dropna().unique().tolist(),
        df["target"].dropna().unique().tolist(),
        source_study_name,
        target_study_name,
        graph_db_repo
        
    )

    
    # move "mapping_relation" to the end
    if "mapping_relation" in df.columns:
        mapping_type = df.pop("mapping_relation")
        df["mapping_relation_older"] = mapping_type
    
    df[["transformation_rule",  "semantic_relationship", "harmonization_status"]]   = df.apply(
        lambda row: apply_rules(
            domain=row.get("category", "") if "category" in row and pd.notna(row.get("category")) else "",
            mapping_relation=row.get("mapping_relation_older", "") if "mapping_relation_older" in row and pd.notna(row.get("mapping_relation_older")) else "",
            
            src_info={
                "var_name": row.get("source") if "source" in row and pd.notna(row.get("source")) else None,
                "omop_id": row.get("somop_id") if "somop_id" in row and pd.notna(row.get("somop_id")) else None,
                "stats_type": row.get("source_type") if "source_type" in row and pd.notna(row.get("source_type")) else None,
                "unit": row.get("source_unit") if "source_unit" in row and pd.notna(row.get("source_unit")) else None,
                "data_type": row.get("source_data_type") if "source_data_type" in row and pd.notna(row.get("source_data_type")) else None,
                "categories_labels": row.get("source_categories_labels") if "source_categories_labels" in row and pd.notna(row.get("source_categories_labels")) else None,
                "original_categories": row.get("source_original_categories") if "source_original_categories" in row and pd.notna(row.get("source_original_categories")) else None,
                "composite_code": row.get("source_composite") if "source_composite" in row and pd.notna(row.get("source_composite")) else None,
                "visit": row.get("source_visit") if "source_visit" in row and pd.notna(row.get("source_visit")) else None
            },
            tgt_info={
                "var_name": row.get("target") if "target" in row and pd.notna(row.get("target")) else None,
                "omop_id": row.get("tomop_id") if "tomop_id" in row and pd.notna(row.get("tomop_id")) else None,
                "stats_type": row.get("target_type") if "target_type" in row and pd.notna(row.get("target_type")) else None,
                "unit": row.get("target_unit") if "target_unit" in row and pd.notna(row.get("target_unit")) else None,
                "data_type": row.get("target_data_type") if "target_data_type" in row and pd.notna(row.get("target_data_type")) else None,
                "categories_labels": row.get("target_categories_labels") if "target_categories_labels" in row and pd.notna(row.get("target_categories_labels")) else None,
                "original_categories": row.get("target_original_categories") if "target_original_categories" in row and pd.notna(row.get("target_original_categories")) else None,
                "composite_code": row.get("target_composite") if "target_composite" in row and pd.notna(row.get("target_composite")) else None,
                "visit": row.get("target_visit") if "target_visit" in row and pd.notna(row.get("target_visit")) else None
            },
        ),
        axis=1,
        result_type="expand",
    )
    
    # delete older mapping_relation column
    if "mapping_relation_older" in df.columns:
        df = df.drop(columns=["mapping_relation_older"])
    # normalize json columns
    df["transformation_rule"] = df["transformation_rule"].apply(
    lambda v: json.dumps(v) if isinstance(v, dict) else v
)
    # move transformation_rule to the last
    if "transformation_rule" in df.columns:
        transformation_rule = df.pop("transformation_rule")
        df["transformation_rule"] = transformation_rule
    # remove semantic_relationship column
    # df = df.drop(columns=["semantic_relationship"])
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            df[col] = df[col].apply(json.dumps)
        elif df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(str)

    return df.drop_duplicates(keep="first")

# def _build_hierarchy_lookup_query(omop_ids: List[int]) -> str:
#     """
#     Constructs a SPARQL query to retrieve all 1-hop related concepts 
#     (parents, children, siblings, equivalents) for a list of input IDs 
#     from the hierarchy graph.
#     """
#     # Create a space-separated string of IDs for the VALUES clause
#     ids_str = " ".join(str(x) for x in omop_ids)
    
#     return f"""
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#     PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
#     PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    
#     SELECT DISTINCT ?start_id ?end_id ?rel_type
#     WHERE {{
#         GRAPH <https://w3id.org/CMEO/graph/hierarchy> {{
#             # 1. Define the URI for the start node
#             VALUES ?raw_id {{ {ids_str} }}
#             BIND(URI(CONCAT("http://omop.org/OMOP/", STR(?raw_id))) AS ?start_uri)
            
#             # 2. Find connections (Bidirectional)
#             {{
#                 ?start_uri rdfs:subClassOf ?end_uri .
#                 BIND("skos:narrowMatch" AS ?rel_type) # Start is child of End
#             }}
#             UNION
#             {{
#                 ?end_uri rdfs:subClassOf ?start_uri .
#                 BIND("skos:broadMatch" AS ?rel_type) # End is child of Start
#             }}
#             UNION
#             {{
#                 ?start_uri skos:has_close_match ?end_uri .
#                 BIND("skos:closeMatch" AS ?rel_type)
#             }}
#             # 3. Extract integer ID from the target URI
#             # We strip the base URL "http://omop.org/OMOP/" (21 chars) to get the ID
#             BIND(STR(?end_uri) AS ?end_uri_str)
#             BIND(xsd:integer(SUBSTR(?end_uri_str, 22)) AS ?end_id)
            
#             # Return the input ID so we can map it back easily
#             BIND(?raw_id AS ?start_id)
#         }}
#     }}
#     """

# def fetch_hierarchy_map(source_ids: List[int]) -> Dict[int, Dict[int, str]]:
#     """
#     Returns a dictionary mapping SourceID -> {TargetID: RelationType}.
#     """
#     hierarchy_map = defaultdict(dict)
    
#     # Chunk IDs to prevent query length issues (e.g., 200 IDs per batch)
#     chunk_size = 200
#     chunks = [source_ids[i:i + chunk_size] for i in range(0, len(source_ids), chunk_size)]
    
#     # We can use ThreadPoolExecutor here too if volume is high
#     for chunk in chunks:
#         if not chunk: continue
#         query = _build_hierarchy_lookup_query(chunk)
#         results = execute_query(query)
        
#         for row in results['results']['bindings']:
#             try:
#                 sid = int(row['start_id']['value'])
#                 tid = int(row['end_id']['value'])
#                 rel = row['rel_type']['value']
#                 hierarchy_map[sid][tid] = rel
#             except (ValueError, KeyError):
#                 continue
                
#     return hierarchy_map
               
# def fetch_variables_statistic_type(var_names_list:list[str], study_name:str, graph_repo: str) -> pd.DataFrame:

#     data_dict = []
#     var_names_list_ = [var_names_list[i:i + 50] for i in range(0, len(var_names_list), 50)]

#     for var_list in var_names_list_:
#         values_str = " ".join(f'"{v}"' for v in var_list)
       
#         query = _build_statistic_query(study_name, values_str, graph_repo)
#         print(f"Executing statistic query for variables: {query}")
#         results = execute_query(query)
#         for result in results["results"]["bindings"]:
#             identifier = result['identifier']['value']
          
#             if identifier in var_names_list:
#                 composite_value = get_var_context(result['code_label']['value']) if 'code_label' in result else None
#                 # print(f"composite value for {identifier}: {composite_value}")
#                 data_dict.append({
#                     'identifier': identifier,
#                     'stat_label': result['stat_label']['value'] if 'stat_label' in result and result['stat_label']['value'].strip() != "" else None,
#                     'unit_label': result['unit_label']['value'] if 'unit_label' in result  and result['unit_label']['value'].strip() != "" else None,
#                     'data_type': result['data_type_val']['value'] if 'data_type_val' in result  and result['data_type_val']['value'].strip() != "" else None,
#                     "categories_labels": result['all_cat_labels']['value'] if 'all_cat_labels' in result  and result['all_cat_labels']['value'].strip() != "" else None,
#                     'original_categories': result['all_original_cat_values']['value'] if 'all_original_cat_values' in result and result['all_original_cat_values']['value'].strip() != "" else None,
#                     'composite':  composite_value
#                 })
#     data_dict = pd.DataFrame.from_dict(data_dict)
#     print(f"head of data dict: {data_dict.head()}")
#     return data_dict




# def fetch_variables_eda(var_names_list:list[str], study_name:str) -> pd.DataFrame:

#     data_dict = []
#     # split var_names_list with 
#     # make multiple lists by having 30 items in each list
#     var_names_list_ = [var_names_list[i:i + 50] for i in range(0, len(var_names_list), 50)]
#     # print(f"length of var_names_list: {len(var_names_list_)}")
#     for var_list in var_names_list_:
#         values_str = " ".join(f'"{v}"' for v in var_list)
#         query = f"""
#             PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#             PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#             PREFIX dc:   <http://purl.org/dc/elements/1.1/>
#             PREFIX obi:  <http://purl.obolibrary.org/obo/obi.owl/>
#             PREFIX cmeo: <https://w3id.org/CMEO/>
#             PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
#             PREFIX iao: <http://purl.obolibrary.org/obo/iao.owl/>
#             PREFIX stato: <http://purl.obolibrary.org/obo/stato.owl/>
#             SELECT DISTINCT
#                 ?identifier
#                 ?stat_label
#                 GROUP_CONCAT(DISTINCT ?statistic_part; separator=";") AS ?all_statistic_parts
#                 GROUP_CONCAT(DISTINCT ?stat_value; separator="; ") AS ?all_stat_values
                

#                 WHERE {{
#                 GRAPH <https://w3id.org/CMEO/graph/{study_name}> {{
                
#                     # Input: dc:identifier values
#                     VALUES ?identifier {{ {values_str}}}

#                     ?dataElement dc:identifier ?identifier .

#                     # Optional: Statistical description
#                     OPTIONAL {{
#                     ?dataElement iao:is_denoted_by ?stat .
#                     ?stat cmeo:has_value ?stat_label.
                    
#                     ?dataset iao:is_about  ?stat.
#                        obi:is_specified_input_of ?eda_process.
#                     ?eda_process a cmeo:exploratory_data_analysis ;
#                         obi:has_specified_output ?eda_output.
                    
#                     ?eda_output a stato:statistic.
                    
                    
#                        OPTIONAL {{

#                                ?eda_output  ro:has_part ?statistic_part.
#                                  ?statistic_part cmeo:has_value ?stat_value.
                           
#                            }}
                    
                    
                      
                    
#                     }}
                   
#                 }}
#                 }}
#                 GROUP BY ?identifier ?stat_label
#                 ORDER BY ?identifier

#         """
#         # print(query)
#         results = execute_query(query)
        
#         for result in results["results"]["bindings"]:
#             identifier = result['identifier']['value']
#             if identifier in var_names_list:
#                 data_dict.append({
#                     'identifier': identifier,
#                     'stat_label': result['stat_label']['value'] if 'stat_label' in result else None,
#                     'unit_label': result['unit_label']['value'] if 'unit_label' in result else None,
#                     'data_type': result['data_type_val']['value'] if 'data_type_val' in result else None,
#                     "categories_labels": result['all_cat_labels']['value'] if 'all_cat_labels' in result else None,
#                     'original_categories': result['all_original_cat_values']['value'] if 'all_original_cat_values' in result else None
#                 })
#     data_dict = pd.DataFrame.from_dict(data_dict)
#     # print(f"head of data dict: {data_dict.head()}")
#     return data_dict



# def _cross_category_matches(
#     source_elements: List[Dict[str, Any]],
#     target_elements: List[Dict[str, Any]],
#     target_study: str,
#     vector_db: Any,
#     embedding_model: Any,
#     collection_name: str,
#     mapping_mode: str,
# ) -> Iterable[Dict[str, Any]]:
#     """Generate cross‑category pairings that share the same ``omop_id`` **and** visit.

#     Parameters
#     ----------
#     source_elements / target_elements
#         Flat lists produced in the first pass of ``build_mappings``.

#     Returns
#     -------
#     List[Dict[str, Any]]
#         New mapping dictionaries labelled ``cross‑category exact match``.
#     """
#     # final: List[Dict[str, Any]] = []
#     CROSS_CATS = {
#         "measurement",
#         "observation",
#         "condition_occurrence",
#         "condition_era",
#         "observation_period",
#     }
#     src_index: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

#     for s in source_elements:
#         s['visit_'] = check_visit_string(s['visit'], s['visit'])
#         src_index[(s["omop_id"], s['visit'])].append(s)

#     for t in target_elements:
#         t['visit_'] = check_visit_string(t['visit'], s['visit'])    
#         key = (t["omop_id"],  t['visit'])
#         for s in src_index.get(key, []):

#             if s['category'].strip().lower() in ["measurement", "observation", "condition_occurrence", "condition_era", "observation_period"] and t['category'].strip().lower() in ["measurement", "observation", "condition_occurrence", "condition_era","observation_period"]:
#                 # tvisit= check_visit_string(t['visit'], visit_constraint)
#                 # svisit = check_visit_string(s['visit'], visit_constraint)
#                 tvisit = check_visit_string(t['visit_'], s['visit_'])
#                 svisit = check_visit_string(s['visit_'], t['visit_'])
#                 # print(f"source visit: {svisit} and target visit: {tvisit}")
#                 # mapping_type = "code match"
#                 if svisit == tvisit:
#                     yield {
#                         "source": s["source"],
#                         "target": t["target"],
#                         "somop_id": s["omop_id"],
#                         "tomop_id": t["omop_id"],
#                         "scode": s["code"],
#                         "slabel": s["code_label"],
#                         "tcode": t["code"],
#                         "tlabel": t["code_label"],
#                         "category": f"{s['category']}", # choose source category over target category |{t['category']}
#                         "mapping_relation": "skos:exactMatch",
#                         "source_visit": s['visit'],
#                         "target_visit":  t['visit'],
#                     }
#     if mapping_mode in {MappingType.OEC.value, MappingType.OED.value, MappingType.OEH.value}:
#     # ALSO CHECK SEMANTIC SIMILARITY ACROSS CATEGORIES IF NO EXACT MATCHES FOUND USING EMBEDDING SEARCH
#         targets_by_omop: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
#         for t in target_elements:
#             targets_by_omop[t["omop_id"]].append(t)

#         # Cache embedding results per (source_label, source_category)
#         embed_cache: Dict[tuple[str, str], set[int]] = {}

#         for s in source_elements:
#             s_label = s["code_label"]
#             s_cat = s["category"].strip().lower()

#             # Only bother for relevant categories
#             if s_cat not in CROSS_CATS:
#                 continue

#             cache_key = (s_label, s_cat)
#             if cache_key in embed_cache:
#                 matched_omops = embed_cache[cache_key]
#             else:
#                 # You had 0.65 as default score; keep that
#                 # score = 0.8
#             # score = 0.65 if s_category in {"drug_exposure", "drug_era"} else 0.8

#                 # We allow matches to any of the CROSS_CATS in the target
#                 matched_omops = search_in_db(
#                         vectordb=vector_db,
#                         embedding_model=embedding_model,
#                         query_text=s_label,
#                         target_study=[target_study],
#                         limit=100,
#                         data_domain=list(CROSS_CATS),
#                         min_score=SIMILARITY_THRESHOLD,
#                         collection_name=collection_name,
#                     )
                
#                 embed_cache[cache_key] = matched_omops

#             if not matched_omops:
#                 continue

#             for omop_id in matched_omops:
#                 for t in targets_by_omop.get(omop_id, []):
#                     t_cat = t["category"].strip().lower()
#                     if t_cat not in CROSS_CATS:
#                         continue

#                     # Visit constraint
#                     svisit = check_visit_string(s["visit"], t["visit"])
#                     tvisit = check_visit_string(t["visit"], s["visit"])
#                     if svisit != tvisit:
#                         continue

#                     yield {
#                         "source": s["source"],
#                         "target": t["target"],
#                         "somop_id": s["omop_id"],
#                         "tomop_id": t["omop_id"],
#                         "scode": s["code"],
#                         "slabel": s["code_label"],
#                         "tcode": t["code"],
#                         "tlabel": t["code_label"],
#                         "category": f"{s['category']}", # choose source category over target category |{t['category']}
#                         "mapping_relation": "skos:relatedMatch",
#                         "source_visit": s["visit"],
#                         "target_visit": t["visit"],
#                     }

# we have symbolic baseline which is ontology_only then we have neural baseline which is embedding_only and its subtypes embedding_only(concepts), embedding_only(description), embedding_only(concepts+description) and then we have hybrid model which is ontology+embedding

# def _graph_vector_matches(
#     src: List[dict[str, Any]],
#     tgt: List[dict[str, Any]],
#     graph: Any,
#     vector_db: Any,
#     embed_model: Any,
#     target_study: str,
#     collection_name: str,
# ) -> List[Dict[str, Any]]:
#     """Match remaining variables using the OMOP graph or embedding search."""
#     final: List[Dict[str, Any]] = []
#     src_map: Dict[tuple, List[dict[str, Any]]] = defaultdict(list)
#     tgt_map: Dict[tuple, List[dict[str, Any]]] = defaultdict(list)

#     for el in src:
#         src_map[(el["omop_id"], el["category"])].append(el)
#     for el in tgt:
#         tgt_map[(el["omop_id"], el["category"])].append(el)

#     unique_targets: Dict[str, set[int]] = {}
#     for el in tgt:
#         unique_targets.setdefault(el["category"], set()).add(el["omop_id"])

#     for (sid, category), s_elems in src_map.items():
#         tgt_ids = unique_targets.get(category, set()) - {sid}
#         if not tgt_ids:
#             continue
#         label = s_elems[0]["code_label"]
#         reachable = None
       
#         reachable = graph.source_to_targets_paths(sid, tgt_ids, max_depth=1, domain=category)
       
#         if reachable:
#             matched = reachable
#         else:
#             # score = 0.65 if category in {"drug_exposure", "drug_era"} else 0.8
#             # score = 0.8
#             matched_db = set(
#                 search_in_db(
#                     vectordb=vector_db,
#                     embedding_model=embed_model,
#                     query_text=label,
#                     target_study=[target_study],
#                     limit=100,
#                     data_domain=[category],
#                     min_score=SIMILARITY_THRESHOLD,
#                     collection_name=collection_name,
#                 )
#             )
#             # add mapping relation info for all matched ids and mapping relation can be relatedMatch
#             matched = set()
#             for tid in matched_db:
#                 matched.add((tid, "skos:relatedMatch"))

#         for tid, mapping_relation in matched:
#             key = (tid, category)
#             if key not in tgt_map:
#                 continue
#             for se in s_elems:
#                 for te in tgt_map[key]:
#                     tv = te['visit']
#                     sv = se['visit']
#                     if se["category"] != te["category"] or check_visit_string(sv, tv) != check_visit_string(tv, sv):
#                         continue
#                     final.append(
#                         {
#                             "source": se.get("source", ""),
#                             "target": te.get("target", ""),
#                             "source_visit": se.get("visit", ""),
#                             "target_visit": te.get("visit", ""),
#                             "somop_id": se.get("omop_id", ""),
#                             "tomop_id": te.get("omop_id", ""),
#                             "scode": se.get("code", ""),
#                             "slabel": se.get("code_label", ""),
#                             "tcode": te.get("code", ""),
#                             "tlabel": te.get("code_label", ""),
#                             "category": category,
#                             "mapping_relation": mapping_relation
#                         }
#                     )
#     return final
