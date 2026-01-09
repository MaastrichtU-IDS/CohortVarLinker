import time
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from typing import Set, List, Tuple, Dict
from omop_graph import OmopGraphNX 
from config import Settings
from rdflib import URIRef
from utils import delete_existing_triples

# --- CONFIGURATION ---
SPARQL_ENDPOINT = Settings().query_endpoint
UPDATE_ENDPOINT = Settings().update_endpoint
TARGET_GRAPH = "https://w3id.org/CMEO/graph/hierarchy"
CSV_PATH = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/concept_relationship_enriched.csv"
PICKLE_PATH = "graph_nx.pkl"

def get_used_omop_ids() -> Set[int]:
    """
    Queries the Triple Store to find all unique OMOP IDs used in source/target graphs.
    """
    print("Step 1: Fetching used OMOP IDs from Triple Store...")
    query = """
    PREFIX cmeo: <https://w3id.org/CMEO/>
    PREFIX iao: <http://purl.obolibrary.org/obo/iao.owl/>
    
    SELECT DISTINCT ?omop_id_val
    WHERE {
        GRAPH ?g {
            ?s iao:denotes ?omop_code.
            ?omop_code a cmeo:omop_id;
                cmeo:has_value ?omop_id_val .
        }
    }
    """
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    ids = set()
    for r in results["results"]["bindings"]:
        try:
            val = r["omop_id_val"]["value"]
            ids.add(int(val))
        except ValueError:
            continue
            
    print(f"Found {len(ids)} unique concepts used in your studies.")
    return ids

def extract_graph_data(omop_graph: OmopGraphNX, used_ids: Set[int]) -> Tuple[List[str], List[str], List[str]]:
    """
    Extracts strictly relevant graph data:
    1. Parents: Only if they group 2+ used variables OR are used variables themselves.
    2. Equivalents: Only if both sides are used variables.
    3. Labels: Only for the nodes that survive the filters.
    """
    print("Step 2: Extracting relevant graph data with strict filtering...")
    
    hierarchy_triples = []
    equivalence_triples = []
    label_triples = []
    
    # Tracking for preventing duplicates
    seen_rels = set()
    nodes_to_label = set()

    # --- PRE-CALCULATION ---
    # To determine if a parent is a "Grouping" concept, we must count how many
    # USED variables point to it.
    parent_counts = defaultdict(set)
    
    print("   Analyzing parentage structure...")
    for uid in used_ids:
        # Get immediate parents (max_depth=1) to find direct groups
        parents = omop_graph.get_all_parents(uid, max_depth=1)
        for p in parents:
            parent_counts[p].add(uid)

    # --- GENERATION ---
    print("   Generating triples...")
    count = 0
    
    for uid in used_ids:
        nodes_to_label.add(uid) # Always label the variables themselves

        # 1. HIERARCHY (Strict Filtering)
        parents = omop_graph.get_all_parents(uid, max_depth=1)
        for p in parents:
            # Condition A: The parent is itself a used variable (Direct Parent-Child in study)
            is_internal_link = p in used_ids
            # print(is_internal_link)
            # Condition B: The parent groups at least 2 used variables (Shared Component)
            is_grouping_node = len(parent_counts[p]) >= 2
            print(f"UID {uid} Parent {p} | Internal: {is_internal_link} | Grouping: {is_grouping_node}")
            if is_internal_link or is_grouping_node:
                pair = (uid, p, "subClassOf")
                if pair not in seen_rels:
                    s = f"<http://omop.org/OMOP/{uid}>"
                    o = f"<http://omop.org/OMOP/{p}>"
                    hierarchy_triples.append(f"{s} rdfs:subClassOf {o} .")
                    seen_rels.add(pair)
                    nodes_to_label.add(p) # Label the parent since we kept it

        # 2. EQUIVALENCE (Strict Filtering)
        equivalents = omop_graph.get_direct_equivalents(uid) 
        for eq in equivalents:
            # Condition: The equivalent concept MUST be in the used_ids set
            # (We only care if Variable A is equivalent to Variable B in our study)
            if eq in used_ids:
                pair = tuple(sorted((uid, eq))) # sort to avoid A->B and B->A duplicates
                if pair not in seen_rels:
                    s = f"<http://omop.org/OMOP/{uid}>"
                    o = f"<http://omop.org/OMOP/{eq}>"
                    # Add bidirectional match
                    equivalence_triples.append(f"{s} skos:has_close_match {o} .")
                    seen_rels.add(pair)
                    nodes_to_label.add(eq) 
        
        count += 1
        if count % 100 == 0:
            print(f"   Processed {count}/{len(used_ids)} variables...", end="\r")

    print(f"\n   Extracting labels for {len(nodes_to_label)} filtered nodes...")
    
    # 3. LABELS
    for node_id in nodes_to_label:
        if node_id in omop_graph.graph:
            node_data = omop_graph.graph.nodes[node_id]
            label = node_data.get("concept_name", "")
            if label:
                # Escape quotes
                clean_label = label.replace('"', '\\"')
                s = f"<http://omop.org/OMOP/{node_id}>"
                label_triples.append(f'{s} rdfs:label "{clean_label}" .')

    print(f"Extracted {len(hierarchy_triples)} hierarchy triples.")
    print(f"Extracted {len(equivalence_triples)} equivalence triples.")
    print(f"Extracted {len(label_triples)} label triples.")
    
    return hierarchy_triples, equivalence_triples, label_triples

def batch_insert(triples: List[str]):
    """
    Generates SPARQL INSERT DATA queries in batches.
    """
    if not triples:
        print("No triples to insert.")
        return

    print(f"Step 3: Inserting {len(triples)} triples into {TARGET_GRAPH}...")
    
    sparql = SPARQLWrapper(UPDATE_ENDPOINT)
    sparql.setMethod(POST)
    
    BATCH_SIZE = 2000 
    
    prefix_str = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX cmeo: <https://w3id.org/CMEO/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    """

    for i in range(0, len(triples), BATCH_SIZE):
        batch = triples[i:i + BATCH_SIZE]
        triples_block = "\n".join(batch)
        
        query = f"""
        {prefix_str}
        INSERT DATA {{ 
            GRAPH <{TARGET_GRAPH}> {{
                {triples_block}
            }} 
        }}
        """
        
        try:
            sparql.setQuery(query)
            sparql.query()
            print(f"Inserted batch {i}-{min(i+BATCH_SIZE, len(triples))}")
        except Exception as e:
            print(f"Error inserting batch {i}: {e}")
   
def main():
    # 1. Load Graph
    delete_existing_triples(TARGET_GRAPH)
    omop_nx = OmopGraphNX(csv_file_path=CSV_PATH, output_file=PICKLE_PATH)
    
    # 2. Get concepts needing hierarchy
    used_ids = get_used_omop_ids()
    
    if not used_ids:
        print("No concepts found via SPARQL. Check connection or data.")
        return

    # 3. Extract data (Hierarchy + Equivalences + Labels)
    hier_triples, eq_triples, lbl_triples = extract_graph_data(omop_nx, used_ids)
    
    # 4. Push to Triple Store (Batched)
    all_triples = hier_triples + eq_triples + lbl_triples
    batch_insert(all_triples)
    
    print("Done! Hierarchy graph is populated (Sparse mode).")

if __name__ == "__main__":
    main()