import pandas as pd
# import cProfile
# import pstats
from src.omop_graph import OmopGraphNX
from src.config import settings
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import json
import os
import glob
import time
from src.variables_kg import process_variables_metadata_file,add_raw_data_graph
from src.study_kg import generate_studies_kg
from src.vector_db import generate_studies_embeddings, search_in_db
from src.utils import (
        get_cohort_mapping_uri,
        delete_existing_triples,
        publish_graph_to_endpoint,
        OntologyNamespaces,
    
    )




from src.fetch import map_source_target



def create_study_metadata_graph(file_path, recreate=False):

    if recreate:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        graph_file_path = f"{base_path}/data/graphs/studies_metadata.trig"
        g=generate_studies_kg(file_path)
        print(f"length of graph: {len(g)}")
        # if isinstance(g, ConjunctiveGraph):
        #     print("Graph is a ConjunctiveGraph with quads.")
        # else:
        #     print("Graph is a standard RDF Graph with triples.")
        # print(OntologyNamespaces.CMEO.value["graph/studies_metadata"])
        if len(g) > 0:
            print(f"delete_existing_triples for graph: {OntologyNamespaces.CMEO.value['graph/studies_metadata']}")
            # delete_existing_triples(f"{settings.sparql_endpoint}/rdf-graphs/studies_metadata")
            delete_existing_triples(graph_uri=OntologyNamespaces.CMEO.value["graph/studies_metadata"])
            response=publish_graph_to_endpoint(g)
            print(f"Metadata graph published to endpoint: {response}")
            g.serialize(destination=graph_file_path, format="trig")
            print(f"Serialized graph to: {graph_file_path}")
            return g
        else:
            print("No metadata found in the file")
            return None
    else:
        print("Recreate flag is set to False. Skipping processing of study metadata.")


def create_cohort_specific_metadata_graph(dir_path, recreate=False):
    # dir_path should be cohort folder which can have each sub-folder as cohort name and each cohort folder should have metadata file with same name as cohort folder
    # g = init_graph()study
    # project directory path

    base_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Base path: {base_path}")
    if  recreate:
        for cohort_folder in os.listdir(dir_path):
            if cohort_folder.startswith('.'):  # Skip hidden files like .DS_Store
                continue
            start_time = time.time()
            cohort_path = os.path.join(dir_path, cohort_folder)
            if os.path.isdir(cohort_path):
                # ➊ Grab every file that ends with .csv, .xlsx or .json
                patterns = ("*.csv", "*.xlsx", "*.json")
                file_candidates: list[str] = []
                for pat in patterns:
                    file_candidates.extend(glob.glob(os.path.join(cohort_path, pat)))
                cohort_metadata_file = None
                eda_file = None
                # ➋ Classify the candidates
                for file in file_candidates:
                    print(f"File: {file}")
                    # Collect *all* metadata spreadsheets
                    if file.lower().endswith((".csv", ".xlsx")):
                        cohort_metadata_file = file
                    # Optionally single out an EDA JSON
                    if os.path.basename(file).lower().startswith("eda") and file.lower().endswith(".json"):
                        eda_file = file
                # print(f"Processing cohort: {cohort_folder} at path: {cohort_path} for metadata file: {cohort_metadata_file}")
                if cohort_metadata_file:
                    if eda_file and os.path.exists(eda_file):
                        print(f"Processing cohort: {cohort_folder} at path: {cohort_path} for eda file: {eda_file}")
                    g, cohort_id = process_variables_metadata_file(cohort_metadata_file, cohort_name=cohort_folder, eda_file_path=eda_file, study_metadata_graph_file_path=f"{base_path}/data/graphs/studies_metadata.trig")
                    if g and len(g) > 0:
                        # print(validate_graph(g))
                        g.serialize(f"{base_path}/data/graphs/{cohort_id}_metadata.trig", format="trig")
                        print(f"Publishing graph for cohort: {cohort_id}")
                  
                        #delete_existing_triples(f"{settings.sparql_endpoint}/rdf-graphs/{cohort_id}")
                        # res=publish_graph_to_endpoint(g, graph_uri=cohort_id) These lines are works with graphDB
                        delete_existing_triples(
                            get_cohort_mapping_uri(cohort_id)
                        )
                        res = publish_graph_to_endpoint(g)
                        print(f"Graph contains {len(g)} triples before serialization.")
                        print(f"Graph published to endpoint: {res} for cohort: {cohort_id}")
                        
                        end_time = time.time()
                        print(f"Time taken to process cohort: {cohort_folder} is: {end_time - start_time}")
                    else:
                        print(f"Error processing metadata file for cohort: {cohort_folder}")
                
            else:
                print(f"Skipping non-directory file: {cohort_folder}")
            print(f"Base path: {base_path}")
    else:
        print("Recreate flag is set to False. Skipping processing of cohort metadata.")

def create_pld_graph(file_path, cohort_name, output_dir=None, recreate=False) -> None:
    if recreate:
        start_time = time.time()
        g=add_raw_data_graph(file_path, cohort_name)
        if len(g) > 0:
            g.serialize(f"{output_dir}/{cohort_name}_pld.trig", format="trig")
            # delete_existing_triples(f"{settings.sparql_endpoint}/rdf-graphs/{cohort_name}_pld")
            # res=publish_graph_to_endpoint(g,graph_uri=f"{cohort_name}_pld")
            
            delete_existing_triples(f"{get_cohort_mapping_uri(cohort_name)}_pld")
            res=publish_graph_to_endpoint(g)

            print(f"Graph published to endpoint: {res} for cohort: graph/{cohort_name}_pld")
            end_time = time.time()
            print(f"Time taken to process PLD: graph/{cohort_name}_pld is: {end_time - start_time}")
        else:
            print("No data found in the file")
    else:
        print("Recreate flag is set to False. Skipping processing of PLD data.")


def check_if_data_exists(endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX omop: <http://omop.org/>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>

    ASK WHERE {
      GRAPH ?graph {
        ?s omop:Has_omop_id ?o .
      }
    }
    """
    
    sparql.setQuery(query)
    results = sparql.query().convert()
    
    return results['boolean']

def list_graphs(endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    sparql.setQuery("""
                    
    SELECT DISTINCT ?graph
    WHERE {
      GRAPH ?graph {
        ?s ?p ?o .
      }
    }
    """)
    results = sparql.query().convert()
    return [result["graph"]["value"] for result in results["results"]["bindings"]]

def get_all_predicates(endpoint_url, graph_uris):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    
    query = """
    SELECT DISTINCT ?p
    WHERE {
      VALUES ?graph { """ + " ".join(f"<{uri}>" for uri in graph_uris) + """ }
      GRAPH ?graph {
        ?s ?p ?o .
      }
    }
    """
    
    sparql.setQuery(query)
    results = sparql.query().convert()
    predicates = [result["p"]["value"] for result in results["results"]["bindings"]]
    print(f"predicates = {predicates}")
    return predicates


def cluster_variables_by_omop(endpoint_url):
    """
    Clusters variables across cohorts based on their omop_id and displays the clusters as a pandas DataFrame.

    :param endpoint_url: str, the SPARQL endpoint URL.
    :return: pd.DataFrame, DataFrame containing omop_id and associated variable names.
    """
    # Check if data exists at the endpoint
    exists = check_if_data_exists(endpoint_url)

    
    if not exists:
        return pd.DataFrame(columns=["omop_id", "variables"])
    
    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    
    # Define the revised SPARQL query
    query = """
    PREFIX cmeo: <https://w3id.org/CMEO/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?omop_id (GROUP_CONCAT(?variable_label; SEPARATOR=", ") AS ?variables)
    WHERE {
      GRAPH ?graph {
        ?variable_uri rdf:type cmeo:variable .
        ?variable_uri cmeo:has_concept ?base_entity .
        ?base_entity omop:has_omop_id ?omop_id .
        ?variable_uri rdfs:label ?variable_label .
      }
    }
    GROUP BY ?omop_id
    HAVING (COUNT(?variable_uri) > 1)
    """
    
    sparql.setQuery(query)
    
    try:
        # Execute the query
        results = sparql.query().convert()
 
    except Exception as e:

        return pd.DataFrame(columns=["omop_id", "variables"])
    
    # Process the results
    clusters = []
    for result in results["results"]["bindings"]:
        omop_id = result["omop_id"]["value"]
        variables_str = result["variables"]["value"]
        variables = [var.strip() for var in variables_str.split(",")]
        clusters.append({"omop_id": omop_id, "variables": variables})

    
    # Convert to pandas DataFrame
    if clusters:
        df = pd.DataFrame(clusters)

    else:
        df = pd.DataFrame(columns=["omop_id", "variables"])
    return df


    
# def search_studyx_elements():


# from owlready2 import get_ontology





def generate_combined_matched_variables(source_study: str, target_studies: list, data_dir: str):
    """
    Generate a combined CSV showing variable mappings across source and target studies.
    Output CSV will have columns as study names and rows as aligned variables.
    """


    output_dir = os.path.join(data_dir, "output")
    first_study = target_studies[0]

    # Start with the first target study mapping
    base_df = pd.read_csv(os.path.join(output_dir, f"{source_study}_{first_study}_full.csv"))
    merged_df = base_df[["source", "target"]].rename(columns={"target": first_study})

    # Merge each subsequent target study
    for tstudy in target_studies[1:]:
        t_df = pd.read_csv(os.path.join(output_dir, f"{source_study}_{tstudy}_full.csv"))
        t_df = t_df[["source", "target"]].rename(columns={"target": tstudy})
        merged_df = pd.merge(merged_df, t_df, on="source", how="outer")

    # Add source study as first column
    merged_df.insert(0, source_study, merged_df["source"])
    merged_df.drop(columns=["source"], inplace=True)

    # Save final file
    output_path = os.path.join(output_dir, f"{source_study}_matched_across_all.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Combined file saved to: {output_path}")




    
def combine_cross_mappings(
    source_study, 
    target_studies, 
    output_dir, 
    combined_output_path
):
    """
    Combines individual study cross-mapping files into a single grouped file by OMOP ID.
    """
    omop_id_tracker = {}
    mapping_dict = {}

    for tstudy in target_studies:
        out_path = os.path.join(output_dir, f'{source_study}_{tstudy}_full.csv')
        df = pd.read_csv(out_path)
        if tstudy not in mapping_dict:
            mapping_dict[tstudy] = {}
        for _, row in df.iterrows():
            src = str(row["source"]).strip()
            tgt = str(row["target"]).strip()
            somop = str(row["somop_id"]).strip()
            tomop = str(row["tomop_id"]).strip()
            slabel = str(row.get("slabel", "")).strip()
            if src not in omop_id_tracker:
                omop_id_tracker[src] = (somop, slabel)
            mapping_dict[tstudy][src] = (tgt, tomop)

    # Group source variables by OMOP ID
    omop_to_source_vars = defaultdict(list)
    for src_var, (somop_id, slabel) in omop_id_tracker.items():
        omop_to_source_vars[somop_id].append(src_var)

    matched_rows = []
    for _, src_vars in omop_to_source_vars.items():
        row = {}
        row[source_study] = ' | '.join(sorted(set(src_vars)))
        for tstudy in target_studies:
            targets = []
            tdict = mapping_dict.get(tstudy, {})
            for src_var in src_vars:
                tgt_pair = tdict.get(src_var)
                if tgt_pair:
                    targets.append(tgt_pair[0])
            row[tstudy] = ' | '.join(sorted(set(targets))) if targets else ''
        matched_rows.append(row)

    final_df = pd.DataFrame(matched_rows)
    final_df.to_csv(combined_output_path, index=False)
    print(f"✅ Combined existing mappings saved to: {combined_output_path}")
    

def combine_all_mappings_to_json(
    source_study, target_studies, output_dir, json_path
):
    # Dict: {source_var: [mapping_dicts]}
    mappings = {}
    for target in target_studies:
        csv_file = os.path.join(output_dir, f"{source_study}_{target}_full.csv")
        if not os.path.exists(csv_file):
            print(f"Skipping {csv_file}, does not exist.")
            continue
        df = pd.read_csv(csv_file)
        for idx, row in df.iterrows():
            # Source variable name
            src_var = str(row["source"]).strip()
            if not src_var:
                continue
            # Initialize dict for this variable if not present
            if src_var not in mappings:
                mappings[src_var] = []
            # Build mapping dict for this target study
            mapping = {"target_study": target}
            # Source columns
            for col in df.columns:
                if col.startswith("source_") or col.startswith("s") or col in [
                    "source", "somop_id", "scode", "slabel",
                    "category", "source_visit", "source_type", "source_unit", "source_data_type"
                ]:
                    mapping[f"s_{col}"] = row[col]
            # Target columns
            for col in df.columns:
                if col.startswith("target_") or col.startswith("t") or col in [
                    "target", "tomop_id", "tcode", "tlabel", 
                    "target_visit", "target_type", "target_unit", "target_data_type"
                ]:
                    mapping[f"{target}_{col}"] = row[col]
            # Extra columns (mapping_type, transformation_rule, etc.)
            for col in df.columns:
                if col not in [
                    "source", "target", "somop_id", "tomop_id",
                    "scode", "slabel", "tcode", "tlabel",
                    "category", "mapping type", "source_visit", "target_visit",
                    "source_type", "target_type", "source_unit", "target_unit",
                    "source_data_type", "target_data_type", "transformation_rule"
                ]:
                    mapping[f"{col}"] = row[col]
            # Always include mapping type and transformation rule
            if "mapping type" in df.columns:
                mapping[f"{target}_mapping_type"] = row["mapping type"]
            if "transformation_rule" in df.columns:
                mapping[f"{target}_transformation_rule"] = row["transformation_rule"]
            # Add to source variable
            mappings[src_var].append(mapping)
    # Compose final JSON dict
    final_json = {}
    for src_var, mapping_list in mappings.items():
        final_json[src_var] = {
            "from": source_study,
            "mappings": mapping_list
        }
    # Save to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"✅ All mappings combined and saved to {json_path}")

# Example usage:
# combine_all_mappings_to_json("time-chf", ["check-hf", "cachexia"], "data/output", "data/output/time-chf_all_mappings.json")

    
# def omop_clean(x):
#         try:
#             return str(int(float(x)))
#         except Exception:
#             return str(x).strip()


# def combine_cross_mappings_v2(
#     source_study,
#     target_studies,
#     output_dir,
#     combined_output_path,
#     extra_columns=None  # List of extra columns to include (optional)
# ):
#     """
#     Combines individual study cross-mapping files into a single grouped file by OMOP ID,
#     including details for both source and target variables in the output.
#     """
#     if extra_columns is None:
#         extra_columns = [
#             "category","mapping_type","source_visit","target_visit",
#             "source_type","source_unit","source_data_type",
#             "target_type","target_unit","target_data_type","transformation_rule"
#         ]

#     omop_id_tracker = {}
#     mapping_dict = {}
#     mapping_details = {}
#     source_details = {}

#     for tstudy in target_studies:
#         out_path = os.path.join(output_dir, f'{source_study}_{tstudy}_full.csv')
#         df = pd.read_csv(out_path)
#         if tstudy not in mapping_dict:
#             mapping_dict[tstudy] = {}
#             mapping_details[tstudy] = {}
#         for _, row in df.iterrows():
#             src = str(row["source"]).strip()
#             tgt = str(row["target"]).strip()
#             somop = str(row["somop_id"]).strip()
#             tomop = str(row["tomop_id"]).strip()
#             slabel = str(row.get("slabel", "")).strip()
#             # Collect mapping info for grouping
#             if src not in omop_id_tracker:
#                 omop_id_tracker[src] = (somop, slabel)
#             mapping_dict[tstudy][src] = (tgt, tomop)
#             # Target variable details
#             tdetail_pieces = []
#             for col in extra_columns:
#                 val = row.get(col, "")
#                 if pd.isna(val):
#                     val = ""
#                 tdetail_pieces.append(f"{col}={val}")
#             mapping_details[tstudy][src] = (tgt, ", ".join(tdetail_pieces))
#             # Source variable details (collected once per unique source var)
#             if src not in source_details:
#                 sdetail_pieces = []
#                 # For each extra column, try "source_{col}" first, fallback to col name
#                 for col in extra_columns:
#                     sval = row.get(f"source_{col}", row.get(col, ""))
#                     if pd.isna(sval):
#                         sval = ""
#                     sdetail_pieces.append(f"{col}={sval}")
#                 source_details[src] = ", ".join(sdetail_pieces)

#     # Group source variables by OMOP ID
#     omop_to_source_vars = defaultdict(list)
#     for src_var, (somop_id, slabel) in omop_id_tracker.items():
#         omop_to_source_vars[somop_id].append(src_var)

#     matched_rows = []
#     for _, src_vars in omop_to_source_vars.items():
#         row = {}
#         # Source study: list source vars with details
#         row[source_study] = ' | '.join(
#             f"{src_var}: {source_details.get(src_var, '')}" for src_var in sorted(set(src_vars))
#         )
#         # Each target: list mapped target vars with details
#         for tstudy in target_studies:
#             tdict = mapping_dict.get(tstudy, {})
#             tdetail = mapping_details.get(tstudy, {})
#             targets = []
#             for src_var in src_vars:
#                 tgt_pair = tdict.get(src_var)
#                 if tgt_pair:
#                     tgt = tgt_pair[0]
#                     detail_str = tdetail.get(src_var, "")
#                     targets.append(f"{tgt}: {detail_str}")
#             row[tstudy] = ' | '.join(targets) if targets else ''
#         matched_rows.append(row)

#     final_df = pd.DataFrame(matched_rows)
#     final_df.to_csv(combined_output_path, index=False)
#     print(f"✅ Combined existing mappings with source and target details saved to: {combined_output_path}")
    
    
def combine_cross_mappings_v3(
    source_study,
    target_studies,
    output_dir,
    combined_output_path,
    extra_columns=None
):
    if extra_columns is None:
        extra_columns = [
            "category","mapping_type","source_visit","target_visit",
            "source_type","source_unit","source_data_type",
            "target_type","target_unit","target_data_type","transformation_rule"
        ]

    omop_id_tracker = {}
    mapping_dict = {}
    mapping_details = {}
    source_details = {}

    for tstudy in target_studies:
        out_path = os.path.join(output_dir, f'{source_study}_{tstudy}_full.csv')
        df = pd.read_csv(out_path)
        if tstudy not in mapping_dict:
            mapping_dict[tstudy] = {}
            mapping_details[tstudy] = {}
        for _, row in df.iterrows():
            src = str(row["source"]).strip()
            tgt = str(row["target"]).strip()
            somop = str(row["somop_id"]).strip()
            tomop = str(row["tomop_id"]).strip()
            slabel = str(row.get("slabel", "")).strip()
            if src not in omop_id_tracker:
                omop_id_tracker[src] = (somop, slabel)
            mapping_dict[tstudy][src] = (tgt, tomop)
            # Target variable details
            tdetail_pieces = []
            for col in extra_columns:
                if col == "transformation_rule":
                    val = row.get(col, "")
                    if pd.isna(val) or val == "":
                        val = ""
                    else:
                        val = f"{src}→{tgt}:{val}"
                    tdetail_pieces.append(f"{col}={val}")
                else:
                    val = row.get(col, "")
                    if pd.isna(val):
                        val = ""
                    tdetail_pieces.append(f"{col}={val}")
            mapping_details[tstudy][src] = (tgt, ", ".join(tdetail_pieces))
            # Source variable details (collected once per unique source var)
            if src not in source_details:
                sdetail_pieces = []
                for col in extra_columns:
                    sval = row.get(f"source_{col}", row.get(col, ""))
                    if pd.isna(sval):
                        sval = ""
                    sdetail_pieces.append(f"{col}={sval}")
                source_details[src] = ", ".join(sdetail_pieces)

    # Group source variables by OMOP ID
    omop_to_source_vars = defaultdict(list)
    for src_var, (somop_id, slabel) in omop_id_tracker.items():
        omop_to_source_vars[somop_id].append(src_var)

    matched_rows = []
    for _, src_vars in omop_to_source_vars.items():
        row = {}
        # Source study: list source vars with details
        row[source_study] = ' | '.join(
            f"{src_var}: {source_details.get(src_var, '')}" for src_var in sorted(set(src_vars))
        )
        # Each target: list mapped target vars with details
        for tstudy in target_studies:
            tdict = mapping_dict.get(tstudy, {})
            tdetail = mapping_details.get(tstudy, {})
            targets = []
            for src_var in src_vars:
                tgt_pair = tdict.get(src_var)
                if tgt_pair:
                    tgt = tgt_pair[0]
                    detail_str = tdetail.get(src_var, "")
                    targets.append(f"{tgt}: {detail_str}")
            row[tstudy] = ' | '.join(targets) if targets else ''
        matched_rows.append(row)

    final_df = pd.DataFrame(matched_rows)
    final_df.to_csv(combined_output_path, index=False)
    print(f"✅ Combined existing mappings with source and target details saved to: {combined_output_path}")
    
if __name__ == '__main__':

    data_dir = 'data'
    cohort_file_path = f"{data_dir}/cohorts"
    cohorts_metadata_file = f"{data_dir}/cohort_metadata_sheet.csv"
    start_time = time.time()
    create_study_metadata_graph(cohorts_metadata_file, recreate=False)
    create_cohort_specific_metadata_graph(cohort_file_path, recreate=False)
    vector_db, embedding_model = generate_studies_embeddings(cohort_file_path, "localhost", "studies_metadata", recreate_db=False)

    source_study = "time-chf"
    target_studies = ["tim-hf", "gissi-hf"]
    
    
    combined_df = None
    omop_id_tracker = {}  # Track source_omop_id per variable
    
    mapping_dict = {}  # {target_study: {source_var: (target_var, target_omop_id)}}
#   The code snippet provided is a Python script that iterates over a list of target studies and
#   performs the following actions for each target study:
    
    # create new outputput based on source studies folder in output directory for each time we ran the script
 
    
    graph = OmopGraphNX(csv_file_path=settings.concepts_file_path)
    for tstudy in target_studies:
        mapping_transformed=map_source_target(source_study_name=source_study, target_study_name=tstudy, 
                                                embedding_model=embedding_model, vector_db=vector_db, 
                                                collection_name="studies_metadata",
                                                graph=graph)

        print(mapping_transformed)
        
        # mapping_transformed = mapping_transformed.drop_duplicates(keep='first') if not mapping_transformed.empty else pd.DataFrame(columns=["source_variable", "target_variable", "source_omop_id", "target_omop_id"])
        mapping_transformed.to_csv(f'{data_dir}/output/{source_study}_{tstudy}_full.csv', index=False)
        
        
    #     if tstudy not in mapping_dict:
    #         mapping_dict[tstudy] = {}
    #     for _, row in mapping_transformed.iterrows():
    #             src = str(row["source"]).strip()
    #             tgt = str(row["target"]).strip()
    #             somop = str(row["somop_id"]).strip()
    #             tomop = str(row["tomop_id"]).strip()
    #             slabel = str(row.get("slabel", "")).strip()
    #             if src not in omop_id_tracker:
    #                 omop_id_tracker[src] = (somop, slabel)
    #             mapping_dict[tstudy][src] = (tgt, tomop)


    # # 1. Group TIME-CHF source variables by OMOP ID
    # omop_to_source_vars = defaultdict(list)
    # for src_var, (somop_id, slabel) in omop_id_tracker.items():
    #     omop_to_source_vars[somop_id].append(src_var)

    # matched_rows = []

    # # 2. For each OMOP ID, build a row: all TIME-CHF vars and all target study matches
    # for _, src_vars in omop_to_source_vars.items():
    #     row = {}
    #     row[source_study] = ' | '.join(sorted(set(src_vars)))
    #     for tstudy in target_studies:
    #         targets = []
    #         tdict = mapping_dict.get(tstudy, {})
    #         for src_var in src_vars:
    #             tgt_pair = tdict.get(src_var)
    #             if tgt_pair:
    #                 targets.append(tgt_pair[0])
    #         row[tstudy] = ' | '.join(sorted(set(targets))) if targets else ''
    #     matched_rows.append(row)

    # # 3. Save the DataFrame
    # final_df = pd.DataFrame(matched_rows)
    # output_path = f'{data_dir}/output/{source_study}_omop_id_grouped_all_targets.csv'
    # final_df.to_csv(output_path, index=False)
    # print(f"✅ Matched variables (grouped by source OMOP ID) saved to: {output_path}")   
    
    tstudy_str = "_".join(target_studies)
    combine_all_mappings_to_json(
        source_study=source_study,
        target_studies=target_studies,
        output_dir=os.path.join(data_dir, "output"),
        json_path=os.path.join(data_dir, "output", f"{source_study}_{tstudy_str}_grouped.json")
    )
    
    # zip all csv files in one folder
    # import zipfile
    # output_dir = os.path.join(data_dir, "output")
    # zip_file_path = os.path.join(output_dir, f"{source_study}_mappings.zip")
    # with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    #     for file in os.listdir(output_dir):
    #         if file.startswith(source_study) and file.endswith(".csv"):
    #             zipf.write(os.path.join(output_dir, file), arcname=file)
    # print(f"✅ All CSV files zipped to: {zip_file_path}")
     
    # omop_to_source_vars = defaultdict(list)
    # for src_var, (somop_id, slabel) in omop_id_tracker.items():
    #     omop_to_source_vars[somop_id].append((src_var, slabel))

    # matched_rows = []
    # for omop_id, src_entries in omop_to_source_vars.items():
    #     # Pick one TIME-CHF variable as representative
    #     src_var, _ = src_entries[0]
    #     row = {source_study: src_var}
    #     found_in_all = True
    #     for tstudy in target_studies:
    #         found = False
    #         for candidate_src, (target_var, target_omop) in mapping_dict.get(tstudy, {}).items():
    #             if str(target_omop).strip() == str(omop_id).strip():
    #                 row[tstudy] = target_var
    #                 found = True
    #                 break
    #         if not found:
    #             found_in_all = False
    #             break
    #     if found_in_all:
    #         matched_rows.append(row)

    # final_df = pd.DataFrame(matched_rows)
    # output_path = f'{data_dir}/output/{source_study}_matched_by_omop_id.csv'
    # final_df.to_csv(output_path, index=False)
    # print(f"✅ Matched variables grouped by OMOP ID saved to: {output_path}")
    #     for _, row in mapping_transformed.iterrows():
    #         src = row["source"]
    #         tgt = row["target"]
    #         omop_id = row["somop_id"]
    #         std_label = row["slabel"]

    #         # Track OMOP ID and label once
    #         if src not in omop_id_tracker:
    #             omop_id_tracker[src] = (omop_id, std_label)

    #         if tstudy not in mapping_dict:
    #             mapping_dict[tstudy] = {}
    #         mapping_dict[tstudy][src] = (tgt, row["tomop_id"])

    #     # Build temp DataFrame
    #     temp_df = mapping_transformed[["source", "target"]].rename(columns={"target": tstudy})
    #     if combined_df is None:
    #         combined_df = temp_df
    #     else:
    #         combined_df = pd.merge(combined_df, temp_df, on="source", how="outer")

    # # Only keep source vars matched across all target studies with consistent OMOP IDs .. var name can be different but 
    # combined_df["valid"] = combined_df["source"].apply(
    #     lambda var: (
    #         var in omop_id_tracker and
    #         all(var in mapping_dict.get(tstudy, {}) for tstudy in target_studies) and
    #         len({omop_id_tracker[var][0]} | {mapping_dict[tstudy][var][1] for tstudy in target_studies}) == 1
    #     )
    # )
    # filtered = combined_df[combined_df["valid"]].copy()

    # # Format each cell: source_var | target_var (standard_label, omop_id)
    # def format_cell(src_var, tgt_var, omop_id):
    #     return f"{src_var} | {tgt_var} ({omop_id})"

    # for tstudy in target_studies:
    #     filtered[tstudy] = filtered["source"].apply(lambda var: format_cell(
    #         var,
    #         mapping_dict[tstudy][var][0],
    #         omop_id_tracker[var][0]
    #     ))

    # # Add source_study as first column
    # filtered.insert(0, source_study, filtered["source"])
    # filtered.drop(columns=["source", "valid"], inplace=True)
    
    # # drop duplicated rows
    # filtered = filtered.drop_duplicates()
    # target_studies_str = "_".join(target_studies)
    # output_path = f'{data_dir}/output/{source_study}_matched_across_{target_studies_str}.csv'
    # filtered.to_csv(output_path, index=False)
    # print(f"✅ Detailed matched table saved to: {output_path}")