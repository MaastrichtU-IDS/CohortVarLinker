from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from functools import lru_cache
from collections import defaultdict
import json
from abc import ABC, abstractmethod
from .config import settings

# --- Configuration & Constants ---
# (Assumed from your .config import settings)
# class Settings:
#     DATE_HINTS = ["visit date", "date of visit", "date of event"]
#     SIMILARITY_THRESHOLD = 0.8
#     GRAPH_REPO = "https://w3id.org/CMEO/graph"
    
# settings = Settings()

# --- 1. THE CONSTRAINT PROGRAMMING ENGINE ---

@dataclass
class CandidateContext:
    """Represents the variables in our Constraint Satisfaction Problem."""
    src: Dict[str, Any]
    tgt: Dict[str, Any]
    
    # Computed properties (Lazy evaluation for constraints)
    @property
    def src_type(self): return str(self.src.get('stats_type', '')).lower().strip()
    @property
    def tgt_type(self): return str(self.tgt.get('stats_type', '')).lower().strip()
    @property
    def src_unit(self): return self.src.get('unit')
    @property
    def tgt_unit(self): return self.tgt.get('unit')
    @property
    def visit_match(self): 
        # Encapsulated Visit Logic
        s_vis = str(self.src.get('visit', '')).lower()
        t_vis = str(self.tgt.get('visit', '')).lower()
        return ConstraintSolver.check_visit_string(s_vis, t_vis) == ConstraintSolver.check_visit_string(t_vis, s_vis)

class Constraint(ABC):
    """Abstract Base Class for a Constraint."""
    @abstractmethod
    def satisfy(self, ctx: CandidateContext) -> Tuple[bool, str, Optional[Dict]]:
        """
        Returns: 
        - bool: Is constraint satisfied? (or is this rule applicable?)
        - str: The resulting harmonization status if applicable.
        - dict: The transformation rule details if applicable.
        """
        pass

# --- 2. DEFINING THE RULES AS CONSTRAINTS ---

class VisitConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        if not ctx.visit_match:
            return False, "Not Applicable", {"description": "Visit time-points are incompatible."}
        return True, None, None # Pass to next constraint

class DataTypeCompatibilityConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        valid_types = {"continuous_variable", "binary_class_variable", "multi_class_variable", "qualitative_variable"}
        
        # Guard Clause (Hard Constraint)
        if ctx.src_type not in valid_types or ctx.tgt_type not in valid_types:
            if 'derived' in str(ctx.src.get('source', '')).lower():
                 return True, "Compatible Match", {"description": "Derived variable requires calculation."}
            return False, "Not Applicable", {"description": "Invalid or missing statistical types."}
        
        return True, None, None

class ContinuousMatchConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        # Only applies if both are continuous
        if ctx.src_type == "continuous_variable" and ctx.tgt_type == "continuous_variable":
            
            # Check Composite Context
            s_comp = ctx.src.get('composite_code')
            t_comp = ctx.tgt.get('composite_code')
            if s_comp != t_comp:
                return True, "Partial Match (Proximate)", {"description": "Different semantic context; manual review required."}
            
            # Check Units
            if ctx.src_unit and ctx.tgt_unit and ctx.src_unit != ctx.tgt_unit:
                return True, "Compatible Match", {"description": f"Unit conversion required: {ctx.src_unit} -> {ctx.tgt_unit}."}
            
            return True, "Identical Match", {"description": "Continuous types and units match."}
        
        return False, None, None # Constraint not applicable to this data type

class CategoricalOverlapConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        # Logic for categorical matching
        if ctx.src_type in ["binary_class_variable", "multi_class_variable", "qualitative_variable"] and \
           ctx.tgt_type == ctx.src_type:
               
            # (Simplified label logic for brevity - utilizing your existing logic concept)
            s_cats = set(str(ctx.src.get('original_categories','')).split(';'))
            t_cats = set(str(ctx.tgt.get('original_categories','')).split(';'))
            
            if s_cats == t_cats:
                 return True, "Identical Match", {"description": "Categories are identical."}
            
            overlap = s_cats.intersection(t_cats)
            if overlap:
                 return True, "Compatible Match", {"description": f"Partial overlap on {len(overlap)} labels."}
            
            return True, "Partial Match (Tentative)", {"description": "No label overlap; mapping required."}

        return False, None, None

class CrossTypeConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        # Logic for mismatched types (e.g. Continuous vs Categorical)
        if ctx.src_type != ctx.tgt_type:
            msg = "Fundamental statistical type mismatch."
            status = "Not Applicable"
            
            # Specific exception logic
            if ctx.src_type == "continuous_variable" and "class" in ctx.tgt_type:
                 msg = "Discretization required (Information Loss)."
                 status = "Partial Match (Tentative)"
            
            return True, status, {"description": msg}
        
        return False, None, None

# --- 3. THE SOLVER ---

class ConstraintSolver:
    def __init__(self):
        # The order of constraints acts as a "Waterfall" or "Sieve"
        
        self.constraints: List[Constraint] = [
            VisitConstraint(),                 # Hard Pruning
            DataTypeCompatibilityConstraint(), # Hard Pruning
            ContinuousMatchConstraint(),       # Specific Rule
            CategoricalOverlapConstraint(),    # Specific Rule
            CrossTypeConstraint()              # Fallback Rule
        ]

    @staticmethod
    @lru_cache(maxsize=None)
    def check_visit_string(visit_str_src: str, visit_str_tgt:str) -> str:
        for hint in settings.DATE_HINTS:
            if hint in visit_str_src.lower(): return visit_str_tgt
            if hint in visit_str_tgt.lower(): return visit_str_src
        return visit_str_src

    def solve(self, src_info: Dict, tgt_info: Dict) -> Tuple[Dict, str]:
        ctx = CandidateContext(src=src_info, tgt=tgt_info)
        
        for constraint in self.constraints:
            is_applicable, status, details = constraint.satisfy(ctx)
            
            # If the constraint returns a Status, it has found a solution (Rule Triggered)
            if status is not None:
                return details, status
            
            # If is_applicable is False (and no status), it means this constraint 
            # hard-failed (e.g. Visit mismatch), so we stop immediately.
            if not is_applicable:
                # If a Hard Constraint fails, we usually return Not Applicable immediately
                # But looking at VisitConstraint, it returns False + Status.
                # If a constraint returns False, None, None, it just means "Rule not applicable, continue"
                continue

        return {"description": "No matching rule found."}, "Not Applicable"

# Initialize Solver Global
harmonization_engine = ConstraintSolver()

# --- 4. DATA PIPELINE (Refactored) ---

class SPARQLQueryBuilder:
    """Responsible solely for constructing valid SPARQL queries."""
    PREFIXES = """
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX sio:  <http://semanticscience.org/ontology/sio.owl/>
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dc:   <http://purl.org/dc/elements/1.1/>
        PREFIX iao:  <http://purl.obolibrary.org/obo/iao.owl/>
        PREFIX cmeo: <https://w3id.org/CMEO/>
        PREFIX obi: <http://purl.obolibrary.org/obo/obi.owl/>
    """
    
    @classmethod
    def build_alignment_query(cls, source: str, target: str, graph_repo: str) -> str:
        return f""" 
            {cls.PREFIXES}
            SELECT ?omop_id
            (SAMPLE(?lbl) AS ?code_label)
            (SAMPLE(?val) AS ?code_value)
            (GROUP_CONCAT(DISTINCT ?src_combined; SEPARATOR=" | ") AS ?source_data)
            (GROUP_CONCAT(DISTINCT ?tgt_combined; SEPARATOR=" | ") AS ?target_data)
            WHERE {{
                {{
                    SELECT ?omop_id ?lbl ?val ?src_combined
                    WHERE {{
                        GRAPH <{graph_repo}/{source}> {{
                             ?deA dc:identifier ?src_var ; skos:has_close_match ?codeSetA .
                             ?codeSetA rdf:_1 ?codeNodeA .
                             ?codeNodeA rdfs:label ?lbl ; cmeo:has_value ?val ; iao:denotes/cmeo:has_value ?omop_id .
                             OPTIONAL {{ ?deA sio:has_attribute/skos:has_close_match/rdfs:label ?src_vis_label . }}
                             BIND(IF(BOUND(?src_vis_label), CONCAT(?src_var, "[", ?src_vis_label, "]"), ?src_var) AS ?src_combined)
                        }}
                    }}
                }}
                UNION
                {{
                     SELECT ?omop_id ?lbl ?val ?tgt_combined
                    WHERE {{
                        GRAPH <{graph_repo}/{target}> {{
                             ?deB dc:identifier ?tgt_var ; skos:has_close_match ?codeSetB .
                             ?codeSetB rdf:_1 ?codeNodeB .
                             ?codeNodeB rdfs:label ?lbl ; cmeo:has_value ?val ; iao:denotes/cmeo:has_value ?omop_id .
                             OPTIONAL {{ ?deB sio:has_attribute/skos:has_close_match/rdfs:label ?tgt_vis_label . }}
                             BIND(IF(BOUND(?tgt_vis_label), CONCAT(?tgt_var, "[", ?tgt_vis_label, "]"), ?tgt_var) AS ?tgt_combined)
                        }}
                    }}
                }}
            }}
            GROUP BY ?omop_id
            ORDER BY ?omop_id
        """

    @classmethod
    def build_statistic_query(cls, source: str, values_str: str, graph_repo: str) -> str:
        return f""" 
            {cls.PREFIXES}
            SELECT DISTINCT ?identifier ?stat_label ?unit_label ?data_type_val
            (COALESCE(?cat_labels_str, "") AS ?all_cat_labels)
            (COALESCE(?cat_values_str, "") AS ?all_original_cat_values)
            WHERE {{
                GRAPH <{graph_repo}/{source}> {{
                   VALUES ?identifier {{ {values_str} }}
                   ?dataElement a cmeo:data_element ; dc:identifier ?identifier .
                   OPTIONAL {{ ?dataElement iao:is_denoted_by/cmeo:has_value ?stat_label . }}
                   OPTIONAL {{
                        {{ ?data_type sio:is_attribute_of ?dataElement . }} UNION {{ ?dataElement sio:has_attribute ?data_type . }}
                        ?data_type a cmeo:data_type ; cmeo:has_value ?data_type_val .
                   }}
                   OPTIONAL {{
                        ?dataElement obi:has_measurement_unit_label ?unitNode .
                        BIND(?unitNode AS ?targetUnit)
                        ?targetUnit cmeo:has_value ?unit_label .
                   }}
                   OPTIONAL {{
                        SELECT ?dataElement 
                        (GROUP_CONCAT(DISTINCT ?catV; SEPARATOR="; ") AS ?cat_values_str)
                        WHERE {{
                            ?cat_val a obi:categorical_value_specification .
                            {{ ?cat_val obi:specifies_value_of ?dataElement . }} UNION {{ ?dataElement obi:has_value_specification ?cat_val . }}
                            ?cat_val cmeo:has_value ?catV .
                        }} GROUP BY ?dataElement
                   }}
                }}
            }}
        """

def _parse_bindings(bindings: Iterable[Dict[str, Any]]) -> tuple[List, List, List]:
    """Return source elements, target elements and exact matches using helper logic."""
    source_elems, target_elems, matches = [], [], []
    
    for result in bindings:
        omop = int(result["omop_id"]["value"])
        code_label = result.get("code_label", {}).get("value", "")
        code_value = result.get("code_value", {}).get("value", "")
        
        # Helper to split "VarName[Visit]"
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

        # Exact Match Generation (Constraint: ID Match AND Visit Match)
        for s, sv in zip(src_vars, src_visits):
            for t, tv in zip(tgt_vars, tgt_visits):
                if ConstraintSolver.check_visit_string(sv, tv) == ConstraintSolver.check_visit_string(tv, sv):
                    matches.append({
                        "source": s, "target": t,
                        "somop_id": omop, "tomop_id": omop,
                        "scode": code_value, "tcode": code_value,
                        "slabel": code_label, "tlabel": code_label,
                        "mapping_relation": "skos:exactMatch",
                        "source_visit": sv, "target_visit": tv
                    })

        # Build Lookups
        def build_el(vars_, visits_, role):
            return [{
                'omop_id': omop, 'code': code_value, 'code_label': code_label,
                'visit': vis, role: el, 'category': 'unknown' # Category populated later if needed
            } for el, vis in zip(vars_, visits_)]

        source_elems.extend(build_el(src_vars, src_visits, "source"))
        target_elems.extend(build_el(tgt_vars, tgt_visits, "target"))
        
    return source_elems, target_elems, matches

def fetch_variables_attributes(var_names_list: list[str], study_name: str, graph_repo: str) -> pd.DataFrame:
    """Fetches statistical metadata in parallel."""
    data_dict = []
    chunk_size = 50
    chunks = [var_names_list[i:i + chunk_size] for i in range(0, len(var_names_list), chunk_size)]

    def process_chunk(var_list):
        chunk_results = []
        try:
            values_str = " ".join(f'"{v}"' for v in var_list)
            query = SPARQLQueryBuilder.build_statistic_query(study_name, values_str, graph_repo)
            # Stub: results = execute_query(query) 
            # Assuming execute_query is imported from utils
            # results = execute_query(query) 
            # For brevity, assuming binding parsing here...
            pass 
        except Exception as e:
            print(f"Error: {e}")
        return chunk_results

    # (Keep existing ThreadPoolExecutor logic from your original code)
    # ...
    return pd.DataFrame(data_dict)

def _check_variable_attributes(df: pd.DataFrame, src_study: str, tgt_study: str, graph_repo: str) -> pd.DataFrame:
    """Attaches statistics to the main dataframe."""
    src_vars = df["source"].dropna().unique().tolist()
    tgt_vars = df["target"].dropna().unique().tolist()
    
    src_stats = fetch_variables_attributes(src_vars, src_study, graph_repo)
    tgt_stats = fetch_variables_attributes(tgt_vars, tgt_study, graph_repo)

    # Merge logic (Preserved from original)
    if not src_stats.empty:
        df = df.merge(src_stats.rename(columns={"identifier": "source", "stat_label": "source_type", "unit_label": "source_unit", "data_type": "source_data_type", "all_original_cat_values": "source_original_categories"}), on="source", how="left")
    if not tgt_stats.empty:
        df = df.merge(tgt_stats.rename(columns={"identifier": "target", "stat_label": "target_type", "unit_label": "target_unit", "data_type": "target_data_type", "all_original_cat_values": "target_original_categories"}), on="target", how="left")
    
    return df

def map_source_target(
    source_study_name: str,
    target_study_name: str,
    vector_db: Any,
    embedding_model: Any,
    graph_db_repo: str = settings.GRAPH_REPO,
    mapping_mode: str = "OO",
) -> pd.DataFrame:
    
    print(f"Building mappings {source_study_name} -> {target_study_name}")
    
    # 1. Fetch Candidates (Constraint: Shared OMOP ID)
    query = SPARQLQueryBuilder.build_alignment_query(source_study_name, target_study_name, graph_db_repo)
    # bindings = execute_query(query) # Stub
    # src_elems, tgt_elems, matches = _parse_bindings(bindings)
    
    # ... (Neurosymbolic Matcher and Derived Variable Logic would be inserted here) ...
    # For this example, we assume `matches` is populated
    matches = [] # Placeholder
    
    df = pd.DataFrame(matches)
    if df.empty: return pd.DataFrame(columns=["source", "target", "harmonization_status"])

    # 2. Enrich with Attributes
    df = _check_variable_attributes(df, source_study_name, target_study_name, graph_db_repo)

    # 3. Apply Constraint Solver (Replacing 'apply_rules')
    print("Applying Constraint Logic...")
    
    def run_solver(row):
        # Extract context from DataFrame row
        src_info = {
            "stats_type": row.get("source_type"),
            "unit": row.get("source_unit"),
            "data_type": row.get("source_data_type"),
            "original_categories": row.get("source_original_categories"),
            "visit": row.get("source_visit")
        }
        tgt_info = {
            "stats_type": row.get("target_type"),
            "unit": row.get("target_unit"),
            "data_type": row.get("target_data_type"),
            "original_categories": row.get("target_original_categories"),
            "visit": row.get("target_visit")
        }
        
        # DELEGATE TO SOLVER
        details, status = harmonization_engine.solve(src_info, tgt_info)
        return json.dumps(details), status

    # Apply row-wise
    df[["transformation_rule", "harmonization_status"]] = df.apply(
        lambda row: run_solver(row), 
        axis=1, 
        result_type="expand"
    )

    return df