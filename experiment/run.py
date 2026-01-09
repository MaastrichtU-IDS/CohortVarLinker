import json
import pandas as pd
from typing import Any, List, Dict
from .config import settings
from .query_builder import SPARQLQueryBuilder
from .constraints import ConstraintSolver, MappingType
from .neuro_matcher import NeuroSymbolicMatcher
from .variable_profile import VariableProfile
from .utils import execute_query

class StudyMapper:
    def __init__(self, vector_db: Any, embed_model: Any, graph: Any = None):
        self.matcher = NeuroSymbolicMatcher(vector_db, embed_model, graph)
        self.solver = ConstraintSolver()

    def _parse_sparql_bindings(self, bindings: List[Dict]) -> tuple[List, List, List]:
        """Convert SPARQL JSON to structured Lists."""
        src_elems, tgt_elems, matches = [], [], []

        for result in bindings:
            omop = int(result["omop_id"]["value"])
            code_label = result.get("code_label", {}).get("value", "")
            code_value = result.get("code_value", {}).get("value", "")
            src_cat = result["source_domain"]["value"].strip().lower()
            tgt_cat = result["target_domain"]["value"].strip().lower()

            def parse_raw(raw_list):
                vars_, visits_ = [], []
                if not raw_list: return vars_, visits_
                for item in raw_list:
                    if "[" in item and item.endswith("]"):
                        parts = item.rsplit("[", 1)
                        vars_.append(parts[0].strip())
                        visits_.append(parts[1].strip(" ]"))
                    else:
                        vars_.append(item.strip())
                        visits_.append("baseline time")
                return vars_, visits_

            src_vars, src_visits = parse_raw(result["source_data"]["value"].split(" | ") if result.get("source_data") else [])
            tgt_vars, tgt_visits = parse_raw(result["target_data"]["value"].split(" | ") if result.get("target_data") else [])

            # Exact Matches (ID (Code/OMOP ID) + Time-point Alignment)
            for s, sv in zip(src_vars, src_visits):
                for t, tv in zip(tgt_vars, tgt_visits):
                    if self.solver.check_visit_string(sv, tv) == self.solver.check_visit_string(tv, sv):
                        category = src_cat if src_cat == tgt_cat else f"{src_cat}|{tgt_cat}"
                        matches.append({
                            "source": s, "target": t,
                            "somop_id": omop, "tomop_id": omop,
                            "scode": code_value, "tcode": code_value,
                            "slabel": code_label, "tlabel": code_label,
                            "mapping_relation": "skos:exactMatch",
                            "source_visit": sv, "target_visit": tv,
                            "category": category
                        })

            # Build Elements for Neurosymbolic Matcher
            def build(vars_, visits_, role, cat):
                return [{
                    "omop_id": omop, "code": code_value, "code_label": code_label,
                    "category": cat, "visit": v, role: n
                } for n, v in zip(vars_, visits_)]

            src_elems.extend(build(src_vars, src_visits, "source", src_cat))
            tgt_elems.extend(build(tgt_vars, tgt_visits, "target", tgt_cat))

        return src_elems, tgt_elems, matches

    def run_pipeline(self, src_study: str, tgt_study: str, 
                     mapping_mode: str = "OO", collection_name: str = "studies") -> pd.DataFrame:
        
        print(f"Aligning {src_study} -> {tgt_study} [{mapping_mode}]")

        # 1. Base Alignment (SPARQL)
        query = SPARQLQueryBuilder.build_alignment_query(src_study, tgt_study, settings.GRAPH_REPO)
        raw = execute_query(query).get("results", {}).get("bindings", [])
        src_elems, tgt_elems, matches = self._parse_sparql_bindings(raw)

        # 2. Neurosymbolic Expansion
        if mapping_mode in {MappingType.OEH.value, MappingType.OEC.value, MappingType.OED.value}:
            ns_matches = self.matcher.resolve_matches(src_elems, tgt_elems, tgt_study, collection_name)
            matches.extend(ns_matches)

        df = pd.DataFrame(matches)
        if df.empty: return pd.DataFrame(columns=["source", "target", "harmonization_status"])
        df = df.drop_duplicates(subset=["source", "target"])

        # 3. Constraint Enrichment (Fetch Stats)
        df = VariableProfile.attach_attributes(df, src_study, tgt_study, settings.GRAPH_REPO)

        # 4. Constraint Solving (Apply Rules)
        def solve_row(row):
            s_info = {
                "stats_type": row.get("source_type"), "unit": row.get("source_unit"),
                "visit": row.get("source_visit"), "original_categories": row.get("source_original_categories")
            }
            t_info = {
                "stats_type": row.get("target_type"), "unit": row.get("target_unit"),
                "visit": row.get("target_visit"), "original_categories": row.get("target_original_categories")
            }
            details, status = self.solver.solve(s_info, t_info)
            return json.dumps(details), status

        df[["transformation_rule", "harmonization_status"]] = df.apply(
            lambda r: solve_row(r), axis=1, result_type="expand"
        )
        
        return df
    
