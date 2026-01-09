# import json
# from pathlib import Path
# import numpy as np

# def load_sigem(emb_path, node2id_path, varmap_path):
#     node2id = json.loads(Path(node2id_path).read_text())
#     id2node = {i: n for n, i in node2id.items()}
#     varname = json.loads(Path(varmap_path).read_text())

#     with open(emb_path, "r", encoding="utf-8") as f:
#         n, d = map(int, f.readline().split())
#         E = np.zeros((n, d), dtype=np.float32)
#         for line in f:
#             parts = line.rstrip("\n").split("\t")
#             idx = int(parts[0])
#             E[idx] = np.array(parts[1:], dtype=np.float32)

#     # normalize rows so cosine similarity = dot product
#     E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

#     return E, node2id, id2node, varname

# def study_of_varnode(varnode: str) -> str:
#     # var::<study>::<iri>
#     return varnode.split("::", 2)[1]

# def topk_matches_across_studies(E, node2id, id2node, varname,
#                                src_study: str, tgt_study: str, k: int = 10):
#     # collect variable nodes
#     var_nodes = [n for n in node2id.keys() if n.startswith("var::")]
#     src_vars = [n for n in var_nodes if study_of_varnode(n) == src_study]
#     tgt_vars = [n for n in var_nodes if study_of_varnode(n) == tgt_study]

#     src_ids = np.array([node2id[n] for n in src_vars], dtype=int)
#     tgt_ids = np.array([node2id[n] for n in tgt_vars], dtype=int)

#     # similarity matrix (|src| x |tgt|)
#     S = E[src_ids] @ E[tgt_ids].T

#     rows = []
#     for i, s_node in enumerate(src_vars):
#         # top-k indices for this source variable
#         idx = np.argpartition(-S[i], kth=min(k, len(tgt_vars)-1))[:k]
#         idx = idx[np.argsort(-S[i][idx])]

#         for rank, j in enumerate(idx, start=1):
#             t_node = tgt_vars[int(j)]
#             rows.append({
#                 "source_study": src_study,
#                 "target_study": tgt_study,
#                 "source_variable": varname.get(s_node, s_node),
#                 "candidate_variable": varname.get(t_node, t_node),
#                 "score": float(S[i][int(j)]),
#                 "rank": rank,
#             })

#     return rows

# E, node2id, id2node, varname = load_sigem(
#     emb_path="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/sigem/sigem_outputCMEO_UNION_SIGEM_IT_5_C_2_GPU_bch_128_lr0012_dim128_Scl_10.emb",
#     node2id_path="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/sigem/node2id.json",
#     varmap_path="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/sigem/varnode_to_human.json",
# )

# rows = topk_matches_across_studies(E, node2id, id2node, varname,
#                                    src_study="time-chf",
#                                    tgt_study="gissi-hf",
#                                    k=10)

# print(rows[:5])



import json
from pathlib import Path
import numpy as np

# -----------------------------
# Load SIGEM outputs
# -----------------------------
def load_sigem(emb_path, node2id_path, varmap_path):
    node2id = json.loads(Path(node2id_path).read_text())
    id2node = {i: n for n, i in node2id.items()}
    varnode_to_human = json.loads(Path(varmap_path).read_text())

    with open(emb_path, "r", encoding="utf-8") as f:
        n, d = map(int, f.readline().split())
        E = np.zeros((n, d), dtype=np.float32)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx = int(parts[0])
            E[idx] = np.array(parts[1:], dtype=np.float32)

    # normalize rows so cosine similarity = dot product
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    return E, node2id, id2node, varnode_to_human


def study_of_varnode(varnode: str) -> str:
    # var::<study>::<iri>
    return varnode.split("::", 2)[1]


# -----------------------------
# Helper: resolve a variable "name" to the actual var:: node in node2id
# -----------------------------
def resolve_varnode(
    node2id: dict,
    varnode_to_human: dict,
    *,
    study: str,
    variable_name: str,
) -> str:
    """
    variable_name can be either:
      - raw identifier like '@6mwt'
      - or full 'time-chf:@6mwt'
    """
    # normalize input
    q = variable_name.strip()

    # if user passed "time-chf:@6mwt", keep only right side for matching
    if ":" in q:
        left, right = q.split(":", 1)
        if left.strip():
            # if they passed a study prefix, prefer it
            study = left.strip()
        q = right.strip()

    # Find all var nodes for this study whose human label ends with ":<q>"
    # human labels look like "time-chf:@6mwt"
    matches = []
    for varnode, human in varnode_to_human.items():
        if not varnode.startswith("var::"):
            continue
        if study_of_varnode(varnode) != study:
            continue
        if human == f"{study}:{q}" or human.endswith(f":{q}"):
            matches.append(varnode)

    if len(matches) == 1:
        return matches[0]

    # fallback: search by substring (useful if human label includes extra text)
    if not matches:
        for varnode, human in varnode_to_human.items():
            if not varnode.startswith("var::"):
                continue
            if study_of_varnode(varnode) != study:
                continue
            if q.lower() in human.lower():
                matches.append(varnode)

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise ValueError(f"Variable '{variable_name}' not found in study '{study}'.")
    else:
        # ambiguous
        raise ValueError(
            f"Variable '{variable_name}' matched multiple nodes in '{study}':\n"
            + "\n".join(varnode_to_human[m] for m in matches[:20])
            + ("\n..." if len(matches) > 20 else "")
        )


# -----------------------------
# Main function: given one variable, return top-5 from another study
# -----------------------------
def topk_similar_variables(
    E: np.ndarray,
    node2id: dict,
    varnode_to_human: dict,
    *,
    src_study: str,
    src_variable_name: str,
    tgt_study: str,
    k: int = 5,
):
    # resolve source varnode
    src_varnode = resolve_varnode(
        node2id, varnode_to_human, study=src_study, variable_name=src_variable_name
    )
    src_id = node2id[src_varnode]

    # target variable nodes
    tgt_varnodes = [
        n for n in node2id.keys()
        if n.startswith("var::") and study_of_varnode(n) == tgt_study
    ]
    if not tgt_varnodes:
        raise ValueError(f"No variable nodes found for target study '{tgt_study}'")

    tgt_ids = np.array([node2id[n] for n in tgt_varnodes], dtype=int)

    # cosine similarities (E is normalized)
    sims = E[src_id] @ E[tgt_ids].T  # (|tgt|,)

    k = min(k, len(tgt_varnodes))
    idx = np.argpartition(-sims, kth=k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    results = []
    for rank, j in enumerate(idx, start=1):
        t_node = tgt_varnodes[int(j)]
        results.append({
            "source_study": src_study,
            "target_study": tgt_study,
            "source_variable": varnode_to_human.get(src_varnode, src_varnode),
            "candidate_variable": varnode_to_human.get(t_node, t_node),
            "score": float(sims[int(j)]),
            "rank": rank,
        })
    return results


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    E, node2id, id2node, varnode_to_human = load_sigem(
        emb_path="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/sigem/sigem_outputCMEO_UNION_SIGEM_IT_5_C_2_GPU_bch_128_lr0012_dim128_Scl_10.emb",
        node2id_path="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/sigem/node2id.json",
        varmap_path="/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/sigem/varnode_to_human.json",
    )

    top5 = topk_similar_variables(
        E, node2id, varnode_to_human,
        src_study="time-chf",
        src_variable_name="V12Nitrat",   # or "time-chf:@6mwt"
        tgt_study="gissi-hf",
        k=10
    )
    print(top5)
    print("\n\n")
    top5 = topk_similar_variables(
        E, node2id, varnode_to_human,
        src_study="time-chf",
        src_variable_name="V12nitrat",   # or "time-chf:@6mwt"
        tgt_study="gissi-hf",
        k=10
    )

    print(top5)
