import networkx as nx
import pandas as pd
import pickle
import os
import gzip
# import zlib 
import time
from typing import List, Tuple
from collections import deque, OrderedDict
from matplotlib import pyplot as plt

class OmopGraphNX:
    def __init__(self, csv_file_path=None, output_file='graph_nx.pkl.gz'):
        self.csv_file_path = csv_file_path
        self.output_file = output_file
        self.graph = nx.DiGraph()
        self._sssp_cache = OrderedDict()
        self._sssp_cache_max = 20000

        if os.path.exists(self.output_file):
            print(f"[INFO] Loading graph from {self.output_file}...")
            start = time.time()
            self.load_graph(self.output_file)
            print(f"[INFO] Graph loaded in {time.time() - start:.2f}s")
        else:
            print("[INFO] Building graph (this may take a moment)...")
            self.build_graph(csv_file_path)

    def build_graph(self, csv_file_path=None):
        csv_file_path = csv_file_path or self.csv_file_path
        if not csv_file_path:
            raise ValueError("No CSV file path provided.")

        print("Reading CSV...")
        # 1. Read Data
        # We need synonym columns now
        use_cols = [
            "concept_id_1", "concept_id_2", "relationship_id",
            "concept_1_vocabulary", "concept_2_vocabulary",
            # "concept_1_domain", "concept_2_domain",
            # "concept_1_concept_class", "concept_2_concept_class",
            "concept_name_1", "concept_name_2",
            # "concept_synonym_1", "concept_synonym_2" 
        ]
        
        # Robust load (only load cols that exist)
        df_header = pd.read_csv(csv_file_path, nrows=0)
        actual_cols = [c for c in use_cols if c in df_header.columns]
        df = pd.read_csv(csv_file_path, usecols=actual_cols, dtype=str)

        # 2. Filter Relationships (Same as before)
        eq_relationships = {
            "rxnorm - atc pr lat": "atc - rxnorm pr lat",
            "atc - rxnorm pr lat": "rxnorm - atc pr lat",
            "atc - rxnorm": "rxnorm - atc",
            "rxnorm - atc": "atc - rxnorm",
            "snomed - rxnorm eq": "rxnorm - snomed eq",
            "rxnorm - snomed eq": "snomed - rxnorm eq",
            'atc - snomed eq': 'snomed - atc eq',
            'snomed - atc eq': 'atc - snomed eq',
            "mapped from": "maps to",
            "maps to": "mapped from",
            # "component of": "has component",
            # "has component": "component of",
        }
        directed = {
            "is a": "subsumes",
            "subsumes": "is a",
            "has answer": "answer of",
            "answer of": "has answer",
        }
        keep_rels = set(eq_relationships.keys()) | set(directed.keys())
        df = df[df["relationship_id"].str.lower().isin(keep_rels)].copy()
        print(f"Rows to process: {len(df):,}")

        # 3. Build Metadata & Synonyms
        # print("Processing Metadata & Compressing Synonyms...")

        # Prepare two dataframes (Stacking concept_1 and concept_2)
        cols_map_1 = {c: c.replace('_1', '') for c in actual_cols if '_1' in c}
        cols_map_2 = {c: c.replace('_2', '') for c in actual_cols if '_2' in c}
        
        df1 = df[list(cols_map_1.keys())].rename(columns=cols_map_1)
        df2 = df[list(cols_map_2.keys())].rename(columns=cols_map_2)
        
        # Stack and Drop Duplicates
        full_meta = pd.concat([df1, df2], ignore_index=True)
        full_meta['concept_id'] = pd.to_numeric(full_meta['concept_id'], errors='coerce').fillna(0).astype('int64')
        full_meta = full_meta[full_meta['concept_id'] != 0]

        # --- OPTIMIZATION A: Standard Metadata (Pandas Category) ---
        # We take the FIRST occurrence of static data (Vocab, Domain, Class, Name)
        meta_df = full_meta.drop_duplicates(subset=['concept_id'])[['concept_id', 'concept_vocabulary', 'concept_name']].copy()
        meta_df.set_index('concept_id', inplace=True)
        
        # Convert to category for compression
        for col in ['concept_vocabulary']:
            meta_df[col] = meta_df[col].fillna("").astype('category')
            
        self.graph.graph['meta'] = meta_df

        # --- OPTIMIZATION B: Synonyms (Zlib Compressed Dictionary) ---
        # Synonyms might be different in different rows for the same ID, so we aggregate them first
        # if 'concept_synonym' in full_meta.columns:
        #     print("Compressing synonym text...")
        #     # 1. Group by ID and join unique synonyms
        #     syn_series = full_meta[['concept_id', 'concept_synonym']].dropna()
        #     # This groupby can be slow. Optimization: Assume pre-aggregated or take first non-null if aggregation is too heavy. 
        #     # If your CSV already has aggregated synonyms (e.g. "aspirin; bufferin"), drop_duplicates is fine.
        #     # If rows have DIFFERENT synonyms, we group:
        #     syn_grouped = syn_series.groupby('concept_id')['concept_synonym'].apply(lambda x: ";".join(set(x.dropna()))).reset_index()
            
        #     # 2. Compress Loop
        #     syn_map_z = {}
        #     for row in syn_grouped.itertuples(index=False):
        #         if row.concept_synonym:
        #             # Compress the string to bytes
        #             syn_map_z[row.concept_id] = zlib.compress(row.concept_synonym.encode('utf-8'))
            
        #     self.graph.graph['syn_map_z'] = syn_map_z
        #     print(f"Compressed {len(syn_map_z):,} synonym entries.")
        
        # 4. Build Graph Structure (Integers Only)
        print("Building Graph Structure...")
        self.graph.add_nodes_from(meta_df.index)
        
        # Map relations to ints
        all_rels = sorted(list(set(eq_relationships) | set(eq_relationships.values()) | set(directed) | set(directed.values())))
        rel_map = {r: i+1 for i, r in enumerate(all_rels)}
        self.graph.graph['rel_map_rev'] = {i: r for r, i in rel_map.items()}

        # Add Edges
        df['u'] = pd.to_numeric(df['concept_id_1'], errors='coerce').fillna(0).astype('int64')
        df['v'] = pd.to_numeric(df['concept_id_2'], errors='coerce').fillna(0).astype('int64')
        
        edges = []
        for row in df.itertuples(index=False):
            rel = row.relationship_id.lower()
            u, v = row.u, row.v
            if u == 0 or v == 0: continue

            r1 = rel_map.get(rel, 0)
            if rel in eq_relationships:
                r2 = rel_map.get(eq_relationships[rel], 0)
                if r1: edges.append((u, v, {'r': r1}))
                if r2: edges.append((v, u, {'r': r2}))
            elif rel in directed:
                if r1: edges.append((u, v, {'r': r1}))
                # Optional: Add inverse if needed
                # r2 = rel_map.get(directed[rel], 0)
                # if r2: edges.append((v, u, {'r': r2}))
        
        self.graph.add_edges_from(edges)
        
        self.save_graph(self.output_file)
        print(f"[INFO] Build Complete.")

    # ------------------------------------------------------------------
    # ACCESSORS (Decompress on fly)
    # ------------------------------------------------------------------
    def get_node_attr(self, node_id, attr):
        try:
            # 1. Check Standard Metadata (Fast)
            if attr in ['vocabulary', 'concept_name', 'name']:
                # Map shorthand
                col = 'concept_' + attr if 'concept' not in attr else attr
                if attr == 'name': col = 'concept_name'
                if attr == 'vocabulary': col = 'concept_vocabulary'
                # if attr == 'domain': col = 'concept_domain'
                # if attr == 'concept_class': col = 'concept_concept_class'

                val = self.graph.graph['meta'].at[node_id, col]
                return str(val) if pd.notna(val) else ""

            # 2. Check Synonyms (Decompress)
            # if attr == 'synonyms' or attr == 'concept_synonym':
            #     z_data = self.graph.graph.get('syn_map_z', {}).get(node_id)
            #     if z_data:
            #         return zlib.decompress(z_data).decode('utf-8')
            #     return ""

        except (KeyError, ValueError, AttributeError, Exception):
            return ""
        return ""
    
    # [Rest of your BFS functions remain exactly the same as previous response]
    # They use get_node_attr, so they will automatically work with the compressed data.
    
    def get_edge_rel(self, u, v):
        if not self.graph.has_edge(u, v): return ""
        edge = self.graph.get_edge_data(u, v)
        r_int = edge.get('r', 0)
        return self.graph.graph.get('rel_map_rev', {}).get(r_int, "")

    def _sssp_lengths(self, start: int, cutoff: int = 3) -> dict:
        key = (start, cutoff)
        hit = self._sssp_cache.get(key)
        if hit:
            self._sssp_cache.move_to_end(key)
            return hit
        dist = dict(nx.single_source_shortest_path_length(self.graph, start, cutoff=cutoff))
        self._sssp_cache[key] = dist
        if len(self._sssp_cache) > self._sssp_cache_max:
            self._sssp_cache.popitem(last=False)
        return dist

    def source_to_targets_paths(self, start, target_ids, max_depth=1):
        try:
            start = int(start)
        except: return []

        if start not in self.graph: return []
        
        vocab_start = self.get_node_attr(start, "vocabulary").lower()

        targets = {int(t) for t in target_ids if int(t) in self.graph and int(t) != start}
        if not targets: return []

        cutoff = max_depth + 3
        dists = self._sssp_lengths(start, cutoff=cutoff)
        results = []

        for tid, dist_edges in dists.items():
            if tid not in targets: continue

            vocab_goal = self.get_node_attr(tid, "vocabulary").lower()
            
            allowed = max_depth
            if vocab_goal not in {"rxnorm", "atc"} and vocab_start not in {"rxnorm", "atc"}:
                allowed = max_depth
            else:
                if vocab_start != vocab_goal:
                    if {vocab_start, vocab_goal} == {"atc", "rxnorm"}: allowed = max_depth + 1
                    elif {vocab_start, vocab_goal} == {"snomed", "atc"}: allowed = max_depth + 2
                    elif {vocab_start, vocab_goal} == {"snomed", "rxnorm"}: allowed = max_depth + 3
                    else: allowed = max_depth + 1
                else:
                    allowed = max_depth + 1 if vocab_start in ("rxnorm", "atc") else max_depth

            if dist_edges > allowed: continue

            match_relation = "skos:relatedMatch"
            if dist_edges == 1:
                rel = self.get_edge_rel(start, tid)
                if "subsumes" in rel: match_relation = "skos:narrowMatch"
                elif "is a" in rel: match_relation = "skos:broadMatch"
                elif "eq" in rel: match_relation = "skos:exactMatch"
            else:
                 if vocab_goal in {"rxnorm", "atc", "snomed"} and vocab_start in {"rxnorm", "atc", "snomed"}:
                    match_relation = "skos:closeMatch"

            results.append((tid, match_relation))
        return results

    # def save_graph(self, pickle_file):
    #     with open(pickle_file, 'wb') as f:
    #         pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"[INFO] Graph saved to {pickle_file}.")
    def save_graph(self, pickle_file):
        # write graph_nx.pkl.gz
        out = pickle_file if pickle_file.endswith(".gz") else pickle_file + ".gz"
        with gzip.open(out, "wb", compresslevel=6) as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Graph saved to {out}")

    # def load_graph(self, pickle_file):
    #     with open(pickle_file, 'rb') as f:
    #         self.graph = pickle.load(f)
    #     print(f"[INFO] Graph loaded. Nodes: {self.graph.number_of_nodes()}")
    def load_graph(self, pickle_file):
        path = pickle_file
        if not path.endswith(".gz"):
            gz = path + ".gz"
            if os.path.exists(gz):
                path = gz
        with gzip.open(path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"[INFO] Graph loaded from {path}. Nodes: {self.graph.number_of_nodes()} Edges: {self.graph.number_of_edges()}")

    def concept_exists(self, concept_id: int, concept_code:str, vocabulary:List[str]) -> Tuple[bool, str]:
        """
        Check if a concept_id exists in the graph.
        """
        # given the concept code and vocabulary check if it exists in the graph
        vocabulary = [v.lower() for v in vocabulary]
        concept_code = concept_code.lower()
        # print(f"value of concept_id is {concept_id}, concept_code is {concept_code} and vocabulary is {vocabulary}")
        if concept_id in self.graph:
            node_data = self.graph.nodes[concept_id]
            # print(f"node data is {node_data}")
            node_vocab = node_data.get("vocabulary", "").strip().lower()
            node_concept_code = node_data.get("concept_code", "").strip().lower()
          
            if node_vocab == '' and node_concept_code == '': 
                return False, "not found"
            elif node_vocab in vocabulary and node_concept_code == concept_code:
                return True, "correct"
            else:
                return False, "incorrect"
    def visualize_graph(self):
        if self.graph.number_of_nodes() == 0: return
        path = list(self.graph.nodes)[:15]
        subgraph = self.graph.subgraph(path)
        pos = nx.spring_layout(subgraph)
        plt.figure(figsize=(12, 8))
        nx.draw(subgraph, pos, with_labels=False, node_size=500, node_color='lightblue', arrows=True)
        
        # Manually fetch labels
        labels = {n: self.get_node_attr(n, 'name') for n in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)
        
        edge_labels = {}
        for u, v, data in subgraph.edges(data=True):
             r_int = data.get('r', 0)
             label = self.graph.graph.get('rel_map_rev', {}).get(r_int, str(r_int))
             edge_labels[(u, v)] = label
             
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Subgraph Visualization")
        plt.savefig("subgraph_visualization.png")

if __name__ == "__main__":
    start_time = time.time()
    csv_path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/concept_relationship_enriched.csv"
    
    omop_nx = OmopGraphNX(csv_path, output_file='graph_nx.pkl.gz')
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")
    
    reachable_targets = omop_nx.source_to_targets_paths(3036277, [37160442], max_depth=1)
    print(f"Reachable targets: {reachable_targets}")