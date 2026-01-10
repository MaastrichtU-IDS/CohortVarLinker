from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from .config import settings
from .vector_db import search_in_db # Stub import
from .constraints import ConstraintSolver

class NeuroSymbolicMatcher:
    def __init__(self, vector_db: Any, embed_model: Any, graph: Any = None):
        self.vector_db = vector_db
        self.embed_model = embed_model
        self.graph = graph

    def resolve_matches(self, src_elements: List[Dict], tgt_elements: List[Dict], 
                       target_study: str, collection_name: str) -> List[Dict]:
        """
        Hybrid Matching: Graph Traversal -> Vector Fallback
        """
        final_matches = []
        
        # 1. Index targets
        tgt_map = defaultdict(list)
        for el in tgt_elements:
            tgt_map[el["omop_id"]].append(el)
        unique_tgt_ids = set(tgt_map.keys())

        # 2. Group sources
        src_grouped = defaultdict(list)
        for el in src_elements:
            src_grouped[el["omop_id"]].append(el)

        for sid, s_group in src_grouped.items():
            candidate_tgt_ids = unique_tgt_ids - {sid}
            if not candidate_tgt_ids: continue
            
            # Representative for searching
            rep = s_group[0]
            
            # A. Symbolic Graph Search
            matched_candidates: Set[Tuple[int, str]] = set()
            if self.graph:
                reachable = self.graph.source_to_targets_paths(
                    sid, candidate_tgt_ids, max_depth=1
                )
                if reachable: matched_candidates = reachable

            # B. Neural Vector Search (Fallback)
            if not matched_candidates:
                raw_matches = search_in_db(
                    vectordb=self.vector_db,
                    embedding_model=self.embed_model,
                    query_text=rep["code_label"],
                    target_study=[target_study],
                    limit=100,
                    # omop_domain=[rep["category"]],
                    min_score=settings.SIMILARITY_THRESHOLD,
                    collection_name=collection_name,
                )
                for tid in raw_matches:
                    matched_candidates.add((tid, "skos:relatedMatch"))
            
            # C. Construct Results
            for tid, relation in matched_candidates:
                if tid not in tgt_map: continue
                
                for tgt in tgt_map[tid]:
                    for src in s_group:
                        # Visit Check
                        s_vis = src['visit']
                        t_vis = tgt['visit']
                        if ConstraintSolver.check_visit_string(s_vis, t_vis) != ConstraintSolver.check_visit_string(t_vis, s_vis):
                            continue
                        
                        final_matches.append({
                            "source": src.get("source"), "target": tgt.get("target"),
                            "source_visit": s_vis, "target_visit": t_vis,
                            "somop_id": src["omop_id"], "tomop_id": tgt["omop_id"],
                            "scode": src.get("code"), "slabel": src.get("code_label"),
                            "tcode": tgt.get("code"), "tlabel": tgt.get("code_label"),
                            "category": src["category"], # Simplified
                            # "mapping_relation": relation
                        })
                        
        return final_matches
    
    