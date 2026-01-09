from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Callable

# import pandas as pd
from pydantic import BaseModel, Field
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm


# -----------------------------
# 1) Your SPARQL query (slightly adjusted: we SELECT ?dataElementA too)
# -----------------------------
def fetch_study_graph_query(study_name: str, graph_repo: str) -> str:
    return f"""
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dc:    <http://purl.org/dc/elements/1.1/>
        PREFIX obi:   <http://purl.obolibrary.org/obo/obi.owl/>
        PREFIX iao:   <http://purl.obolibrary.org/obo/iao.owl/>
        PREFIX cmeo:  <https://w3id.org/CMEO/>

        SELECT
            ("{study_name}" AS ?study_name)
            ?dataElementA
            ?var_nameA
            ?stat_type_value
            ?data_type_value
            ?category
            ?unit_label
            ?categoryies_label
            ?categoryies_value
            ?visit
            ?code_value
            ?code_label
            ?omop_id
        WHERE {{

          GRAPH <{graph_repo}/{study_name}> {{

            ?dataElementA a cmeo:data_element ;
                          dc:identifier ?var_nameA .

            OPTIONAL {{
              ?dataElementA iao:is_denoted_by ?stat .
              ?stat cmeo:has_value ?stat_type_value .
            }}

            OPTIONAL {{
              ?data_type_class a cmeo:data_type ;
                               iao:is_about ?dataElementA ;
                               cmeo:has_value ?data_type_value .
            }}

            OPTIONAL {{
              ?catProcessA a cmeo:categorization_process ;
                           obi:has_specified_input ?dataElementA ;
                           obi:has_specified_output ?catOutA .
              ?catOutA cmeo:has_value ?category .
            }}

            OPTIONAL {{
              ?vd a cmeo:visit_measurement_datum ;
                  iao:is_about ?dataElementA ;
                  obi:is_specified_input_of ?vp .
              ?vp obi:has_specified_output ?vc .
              ?vc rdfs:label ?visit .
            }}
          }}

          OPTIONAL {{
            SELECT ?dataElementA
              (GROUP_CONCAT(DISTINCT ?cv;      SEPARATOR="|") AS ?code_value)
              (GROUP_CONCAT(DISTINCT ?cl;      SEPARATOR="|") AS ?code_label)
              (GROUP_CONCAT(DISTINCT ?oid_str; SEPARATOR="|") AS ?omop_id)
            WHERE {{
              GRAPH <{graph_repo}/{study_name}> {{
                ?std a cmeo:data_standardization ;
                     obi:has_specified_input  ?dataElementA ;
                     obi:has_specified_output ?codeSet .

                ?codeSet ?p ?codeNode .
                FILTER(isIRI(?p) && STRSTARTS(STR(?p), STR(rdf:_)))

                ?codeNode a cmeo:code ;
                          cmeo:has_value ?cv ;
                          rdfs:label ?cl ;
                          iao:denotes ?omopClass .
                ?omopClass a cmeo:omop_id ;
                          cmeo:has_value ?oid .
                BIND(STR(?oid) AS ?oid_str)
              }}
            }}
            GROUP BY ?dataElementA
          }}

          OPTIONAL {{
            SELECT ?dataElementA (SAMPLE(?ul) AS ?unit_label)
            WHERE {{
              GRAPH <{graph_repo}/{study_name}> {{
                ?dataElementA obi:has_measurement_unit_label ?unit .
                ?unit obi:is_specified_input_of ?mu_std .
                ?mu_std obi:has_specified_output ?unit_code .
                ?unit_code cmeo:has_value ?ul .
              }}
            }}
            GROUP BY ?dataElementA
          }}

          OPTIONAL {{
            SELECT ?dataElementA
              (GROUP_CONCAT(DISTINCT ?lbl;  SEPARATOR="|") AS ?categoryies_label)
              (GROUP_CONCAT(DISTINCT ?orig; SEPARATOR="|") AS ?categoryies_value)
            WHERE {{
              GRAPH <{graph_repo}/{study_name}> {{
                ?cat_val a obi:categorical_value_specification ;
                         obi:specifies_value_of ?dataElementA ;
                         obi:is_specified_input_of ?mu_standardization_c ;
                         cmeo:has_value ?orig .
                ?mu_standardization_c obi:has_specified_output ?cat_codes .
                ?cat_codes rdfs:label ?lbl .
              }}
            }}
            GROUP BY ?dataElementA
          }}
        }}
    """


# -----------------------------
# 2) Pydantic models to store study variable profiles
# -----------------------------
def _split_pipe(v: Optional[str]) -> List[str]:
    if not v:
        return []
    return [x.strip() for x in v.split("|") if x.strip()]


class VariableProfile(BaseModel):
    study: str
    data_element_iri: str
    var_name: str

    stat_types: str = Field(default_factory=str)
    data_types: str = Field(default_factory=str)
    domains: str = Field(default_factory=str)
    units: str = Field(default_factory=str)
    visits: str = Field(default_factory=str)

    code_values: List[str] = Field(default_factory=list)
    code_labels: List[str] = Field(default_factory=list)
    omop_ids: List[str] = Field(default_factory=list)

    cat_labels: List[str] = Field(default_factory=list)
    cat_values: List[str] = Field(default_factory=list)

    @property
    def var_node(self) -> str:
        # variable node must be unique per study; keep it deterministic
        # (use data element IRI + study to be safe)
        return f"var::{self.study}::{self.data_element_iri}"


class StudyGraph(BaseModel):
    study: str
    variables: Dict[str, VariableProfile] = Field(default_factory=dict)  # key = data_element_iri


# -----------------------------
# 3) SPARQL utilities
# -----------------------------
def sparql_select(endpoint_url: str, query: str, timeout_s: int = 120) -> List[Dict[str, str]]:
    sp = SPARQLWrapper(endpoint_url)
    sp.setQuery(query)
    sp.setReturnFormat(JSON)
    sp.setTimeout(timeout_s)
    res = sp.query().convert()
    rows: List[Dict[str, str]] = []
    for b in res["results"]["bindings"]:
        row = {}
        for k, v in b.items():
            row[k] = v.get("value")
        rows.append(row)
    return rows


def build_study_graph(endpoint_url: str, graph_repo: str, study: str) -> StudyGraph:
    q = fetch_study_graph_query(study, graph_repo)
    rows = sparql_select(endpoint_url, q)

    sg = StudyGraph(study=study)

    for r in rows:
        de = r.get("dataElementA")
        if not de:
            continue
        var = sg.variables.get(de)
        if var is None:
            var = VariableProfile(
                study=study,
                data_element_iri=de,
                var_name=r.get("var_nameA") or de,
            )
            sg.variables[de] = var

        if r.get("stat_type_value"):
            var.stat_types = r["stat_type_value"]
        if r.get("data_type_value"):
            var.data_types = r["data_type_value"]
        if r.get("category"):
            var.domains = r["category"]
        if r.get("unit_label"):
            var.units = r["unit_label"]
        if r.get("visit"):
            var.visits = r["visit"]
        for x in _split_pipe(r.get("code_value")):
            var.code_values.append(x)
        for x in _split_pipe(r.get("code_label")):
            var.code_labels.append(x)
        for x in _split_pipe(r.get("omop_id")):
            var.omop_ids.append(x)

        for x in _split_pipe(r.get("categoryies_label")):
            var.cat_labels.append(x)
        for x in _split_pipe(r.get("categoryies_value")):
            var.cat_values.append(x)

    return sg


# -----------------------------
# 4) Build the UNION graph edges (typed edges via predicate nodes)
# -----------------------------
def _pnode(p: str) -> str:
    return f"p::{p}"

def add_edge(edges: Set[Tuple[str, str]], a: str, b: str, undirected: bool = True) -> None:
    edges.add((a, b))
    if undirected:
        edges.add((b, a))
        
def _vnode(kind: str, value: str) -> str:
    # normalize a bit to reduce accidental duplicates
    value = re.sub(r"\s+", " ", value.strip())
    return f"{kind}::{value}"

def add_typed_edge(edges: Set[Tuple[str, str]], src: str, pred: str, obj: str, undirected: bool = True) -> None:
    # Represent a typed edge as: src -> p::pred -> obj
    pn = _pnode(pred)
    edges.add((src, pn))
    edges.add((pn, obj))
    if undirected:
        edges.add((pn, src))
        edges.add((obj, pn))


def build_union_edges(
    studies: List[StudyGraph],
    undirected: bool = True,
    *,
    include_domains: bool = False,          # domains are often generic hubs; default False
    include_cat_labels: bool = True,       # labels can be noisy hubs; default False
    include_cat_values: bool = False,        # values are useful if namespaced via value-set node
    cat_value_parent_fn: Optional[Callable[[str], List[str]]] = None,
    # cat_value_parent_fn: given a value concept node id (e.g., "catomop::123"), returns parent concept node ids
) -> Tuple[Set[Tuple[str, str]], Dict[str, str]]:
    """
    Graph schema (SIGEM-friendly):
      var::<study>::<iri>  -- connected directly to feature nodes:
        unit::<...>, visit::<...>, omop::<...>, code::<...>, stat::<...>, dtype::<...>, domain::<...>

      For categorical vars:
        var  <->  pvs::<study>::<iri>  <->  cat::<value>  (or catlabel::<...>)
      Optional hierarchy:
        cat::<child> <-> cat::<parent>   (subsumption / is-a like)
    """
    edges: Set[Tuple[str, str]] = set()
    varnode_to_human: Dict[str, str] = {}

    for sg in studies:
        for vp in sg.variables.values():
            v = vp.var_node
            varnode_to_human[v] = f"{sg.study}:{vp.var_name}"

            # ---- Strong discriminators (good bridges) ----
            # if vp.units:
            #     add_edge(edges, v, _vnode("unit", vp.units), undirected)
            # if vp.visits:
            #     add_edge(edges, v, _vnode("visit", vp.visits), undirected)
            # if vp.stat_types:
            #     add_edge(edges, v, _vnode("stat", vp.stat_types), undirected)
            # if vp.data_types:
            #     add_edge(edges, v, _vnode("dtype", vp.data_types), undirected)

            # OMOP / codes: strongest cross-study bridges
            # for oid in vp.omop_ids:
            #     add_edge(edges, v, _vnode("omop", oid), undirected)
            for cv in vp.code_labels:
                add_edge(edges, v, _vnode("code", cv), undirected)

            # ---- Weak / hub-prone features (use sparingly) ----
            # if include_domains and vp.domains:
            #     add_edge(edges, v, _vnode("domain", vp.domains), undirected)

            # ---- Permissible values as a VALUE-SET node (prevents global yes/no hubs) ----
            if include_cat_values and (vp.cat_values or (include_cat_labels and vp.cat_labels)):
                pvs = f"pvs::{vp.study}::{vp.data_element_iri}"  # per-variable value-set node (NOT a global hub)
                add_edge(edges, v, pvs, undirected)

                # values: namespace them via pvs node (so "yes" doesn't connect all variables)
                for ov in vp.cat_values:
                    # if you later have OMOP ids per value, prefer: catomop::<id>
                    val_node = _vnode("cat", f"{vp.var_name}::{ov}")  # namespaced to avoid hubs
                    add_edge(edges, pvs, val_node, undirected)

                    # Optional: add subsumption/hierarchy edges if you can provide parents
                    if cat_value_parent_fn is not None:
                        for parent in cat_value_parent_fn(val_node):
                            add_edge(edges, val_node, parent, undirected)

                if include_cat_labels:
                    for lbl in vp.cat_labels:
                        lbl_node = _vnode("catlabel", f"{vp.var_name}::{lbl}")
                        add_edge(edges, pvs, lbl_node, undirected)

    return edges, varnode_to_human


# -----------------------------
# 5) Write edge list with 0-indexed ints (SIGEM format)
# -----------------------------
def write_sigem_edgelist(
    edges: Iterable[Tuple[str, str]],
    out_graph_path: Path,
    out_node_map_path: Path,
) -> Dict[str, int]:
    nodes: Set[str] = set()
    edge_list = list(edges)
    for s, t in edge_list:
        nodes.add(s)
        nodes.add(t)

    node2id = {n: i for i, n in enumerate(sorted(nodes))}
    # out_graph_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_graph_path, "w", encoding="utf-8") as f:
        for s, t in edge_list:
            f.write(f"{node2id[s]}\t{node2id[t]}\n")

    with open(out_node_map_path, "w", encoding="utf-8") as f:
        json.dump(node2id, f, indent=2)

    return node2id


# -----------------------------
# 6) Train SIGEM (calls their SIGEM.py CLI)
# -----------------------------
def train_sigem(
    sigem_repo_dir: Path,
    graph_file: Path,
    dataset_name: str,
    result_dir: Path,
    *,
    dim: int = 128,
    itr: int = 5,
    damping_factor: float = 0.2,
    scaling_factor: int = 2,
    epc: int = 100,
    lr: float = 0.0012,
    gpu: bool = True,
    bch_gpu: int = 128,
    bch_cpu: int = 128,
    prl_num: int = 8,
) -> None:
    """
    SIGEM CLI parameters are documented in the repo README. :contentReference[oaicite:2]{index=2}
    """
    # result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "/Users/komalgilani/Documents/GitHub/CohortVarLinker/experiment/SIGEM.py",
        "--graph", str(graph_file),
        "--dataset_name", dataset_name,
        "--result_dir", str(result_dir),
        "--dim", str(dim),
        "--itr", str(itr),
        "--damping_factor", str(damping_factor),
        "--scaling_factor", str(scaling_factor),
        "--gpu", str(gpu),
        "--bch_gpu", str(bch_gpu),
        "--bch_cpu", str(bch_cpu),
        "--prl_num", str(prl_num),
        "--epc", str(epc),
        "--lr", str(lr),
        "--early_stop", "True",
        "--wait_thr", "10",
    ]
    print("\n[RUN]", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


# -----------------------------
# 7) Main
# -----------------------------
def main():
    # ---- EDIT THESE ----
    endpoint_url = "http://localhost:7879/query"   # your Oxigraph/SPARQL endpoint
    graph_repo = "https://w3id.org/CMEO/graph"       # base IRI you used in GRAPH <{graph_repo}/{study_name}>
    study_names = ["time-chf", "gissi-hf", "aachen-hf", "gissi-hf_outcome"]
    
    
    # where SIGEM repo lives
    sigem_repo_dir = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/sigem"

    out_dir = sigem_repo_dir
    graph_file = f"{out_dir}/union_graph.txt"
    node_map_file =f"{out_dir}/node2id.json"
    var_map_file = f"{out_dir}/varnode_to_human.json"
    
    
    dataset_name = "CMEO_UNION"
    result_dir = f"{out_dir}/sigem_output"
    # out_dir.mkdir(parents=True, exist_ok=True)

    
    # 1) fetch each study
    study_graphs: List[StudyGraph] = []
    for s in tqdm(study_names, desc="Fetching named graphs"):
        sg = build_study_graph(endpoint_url, graph_repo, s)
        study_graphs.append(sg)

    # 2) build union edges
    edges, varnode_to_human = build_union_edges(study_graphs, undirected=True)

    # 3) write SIGEM edge list + mappings
    node2id = write_sigem_edgelist(edges, graph_file, node_map_file)
    with open(var_map_file, "w", encoding="utf-8") as f:
        json.dump(varnode_to_human, f, indent=2)

    print(f"[OK] Wrote union edge list: {graph_file}")
    print(f"[OK] Node map: {node_map_file}")
    print(f"[OK] Variable labels: {var_map_file}")
    print(f"[INFO] #nodes={len(node2id)}  #edges={len(edges)}")

    # 4) train SIGEM
    train_sigem(
        sigem_repo_dir=sigem_repo_dir,
        graph_file=graph_file,
        dataset_name=dataset_name,
        result_dir=result_dir,
        dim=128,
        itr=5,
        damping_factor=0.5,
        scaling_factor=2,
        epc=100,
        lr=0.0012,
        gpu=True,
        bch_gpu=128,
    )

    print(f"\n[DONE] SIGEM finished. Check: {result_dir}\n")


if __name__ == "__main__":
    main()
