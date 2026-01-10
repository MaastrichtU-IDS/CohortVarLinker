from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, List, Optional
from functools import lru_cache
from .config import settings
from .lazy_model import similarity_score
from enum import Enum

class MappingType(str, Enum):
    OO = "ontology_only"
    OED = "ontology+embedding(description)"
    OEC = "ontology+embedding(concept)"
    OEH = "ontology+embedding(hybrid)"


class EmbeddingType(str, Enum):
    ED = "embedding(description)"
    EC = "embedding(concept)"
    EH = "embedding(hybrid)"

class DATATYPE(str, Enum):
    CONTINUOUS = "continuous_variable"
    BINARY = "binary_class_variable"
    MULTI_CLASS = "multi_class_variable"
    QUALITATIVE = "qualitative_variable"
    DERIVED = "derived_variable"

@dataclass
class CandidateContext:
    """The variables available to the Constraint Solver."""
    src: Dict[str, Any]
    tgt: Dict[str, Any]
    
    @property
    def src_type(self) -> str: return str(self.src.get('stats_type', '')).lower().strip()
    @property
    def tgt_type(self) -> str: return str(self.tgt.get('stats_type', '')).lower().strip()
    @property
    def src_unit(self) -> str: return self.src.get('unit')
    @property
    def tgt_unit(self) -> str: return self.tgt.get('unit')
    
    @property
    def visit_match(self) -> bool:
        s_vis = str(self.src.get('visit', '')).lower()
        t_vis = str(self.tgt.get('visit', '')).lower()
        return ConstraintSolver.check_visit_string(s_vis, t_vis) == ConstraintSolver.check_visit_string(t_vis, s_vis)

class Constraint(ABC):
    @abstractmethod
    def satisfy(self, ctx: CandidateContext) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Returns:
            - is_applicable (bool): Should we continue checking other constraints?
            - status (str): The final status if matched (e.g. "Identical Match"). None if not final.
            - details (dict): The transformation rule or error description.
        """
        pass

# --- Constraints Implementation ---

class VisitConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        visit_match = ctx.visit_match
        if visit_match:
            if ctx.src_type != ctx.tgt_type:
                return False, "Compatible Match", {"description": "Time-point alignment requires manual review."}
            else:
                return True, None, None
        else:
            return False, "Not Applicable", {"description": "Visit time-points are incompatible."}
       

class DataTypeConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        valid = {DATATYPE.CONTINUOUS.value, DATATYPE.BINARY.value, DATATYPE.MULTI_CLASS.value, DATATYPE.QUALITATIVE.value, DATATYPE.DERIVED.value}
        
        # Check if types are known and valid
        if ctx.src_type not in valid or ctx.tgt_type not in valid:
            # Exception for derived variables which might not have stats yet
            if 'derived' in str(ctx.src.get('source', '')).lower() or 'derived' in str(ctx.tgt.get('target', '')).lower():
                 return True, "Compatible Match", {"description": "Derived variable requires calculation."}
            return False, "Not Applicable", {"description": "Invalid or missing statistical types."}
        
        return True, None, None

class UnitLogicConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        if ctx.src_type == "continuous_variable" and ctx.tgt_type == "continuous_variable":
            
            # 1. Check Semantics (Composite Codes)
            # s_comp = ctx.src.get('composite_code')
            # t_comp = ctx.tgt.get('composite_code')
            # if s_comp != t_comp:
            #     return True, "Partial Match (Proximate)", {"description": "Different semantic context; manual review required."}
            
            # 2. Check Units
            if ctx.src_unit and ctx.tgt_unit and ctx.src_unit != ctx.tgt_unit:
                return True, "Compatible Match", {"description": f"Unit conversion required: {ctx.src_unit} -> {ctx.tgt_unit}."}
            
            return True, "Identical Match", {"description": "Continuous types and units match."}
        
        return True, None, None # Pass if not continuous

class ValueSetLogicConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        if ctx.src_type in {DATATYPE.BINARY.value, DATATYPE.MULTI_CLASS.value}:
            s_raw = str(ctx.src.get('original_categories', ''))
            t_raw = str(ctx.tgt.get('original_categories', ''))
            s_cats = set(x.strip() for x in s_raw.split(';') if x.strip())
            t_cats = set(x.strip() for x in t_raw.split(';') if x.strip())
            if ctx.tgt_type == ctx.src_type:
              
                if not s_cats or not t_cats:
                    return True, "Partial Match (Tentative)", {"description": "Missing categorical values for comparison."}

                if s_cats == t_cats:
                    if s_raw == t_raw:
                        return True, "Identical Match", {"description": "Categorical values labels are identical."}
                    else:
                        return True, "Compatible Match", {"description": "Original categorical values differs in formatting."}
                
                overlap = s_cats.intersection(t_cats)
                if overlap:
                    return True, "Compatible Match", {"description": f"Partial overlap in {len(overlap)} categorical values labels."}
                
                return True, "Partial Match (Tentative)", {"description": "No category value label overlap; mapping required."}
            else:
                # what one is binary and the other is multi-class or vice versa and one is subset of other
                if (ctx.src_type == DATATYPE.BINARY.value and s_cats.issubset(t_cats)) or \
                   (ctx.tgt_type == DATATYPE.BINARY.value and t_cats.issubset(s_cats)):
                    return True, "Compatible Match", {"description": "Binary categories are subset of multi-class categories."} 
                return True, "Not Applicable", {"description": "Values mismatch for categorical variables."}
                
                
        # if src type and tgt type are different
        return True, None, None

class ContextualSimilarityConstraint(Constraint):
    def __init__(self):
        self.threshold = 0.85
        # self.hard_cutoff = 0.50  # Below this, we reject immediately

    def satisfy(self, ctx: CandidateContext):
        src_code_str = ctx.src.get("context_labels")
        tgt_code_str = ctx.tgt.get("context_labels")

        # 1. Pass if missing context (Let type constraints handle it)
        if not src_code_str and not tgt_code_str:
            return True, None, None

        # [ ... Clean text code ... ]
        def clean_composite(s): 
            if not s: return ""
            return " ".join(s.replace("|", " ").split())
        
        text_a = clean_composite(src_code_str)
        text_b = clean_composite(tgt_code_str)

        # 2. Compute Score
        score = similarity_score(settings.EMBEDDING_MODEL_NAME, text_a, text_b)
        
        # 3. Store score in context for later constraints to see
        # (You need to add this field to your CandidateContext dataclass)
        ctx.semantic_score = score 

        # 4. LOGIC CHANGE:
        if score < self.threshold:
            # STOP: The meaning is too different.
            return False, "Not Applicable", {
                "description": f"Context mismatch (Score: {score:.2f})."
            }
        
        # CONTINUE: The meaning matches well enough. 
        # We return None so the solver checks Data Types and Units next.
        return True, None, {"similarity_score": score}
# class ContextualSimilarityConstraint(Constraint):
#     def __init__(self):
#        self.threshold = 0.85
#     def _get_status_from_score(self, score: float, is_visit_match: bool) -> Tuple[str, str]:
#         """Determines status based on Similarity Score and Temporal Match."""
#         # 1. Base Status based on Semantic Context
#         if score >= 0.90:
#             base_status = "Identical Match"
#             desc = "Contexts are semantically identical."
#         elif score >= self.threshold:
#             base_status = "Compatible Match"
#             desc = f"Contexts are semantically similar (Score: {score:.2f})."
#         else:
#             base_status = "Partial Match (Tentative)"
#             desc = f"Contexts differ significantly (Score: {score:.2f}); manual verification required."

#         # 2. Downgrade based on Temporal Mismatch
#         if not is_visit_match:
#             # Downgrade logic (Identical -> Compatible -> Partial)
#             if base_status == "Identical Match":
#                 base_status = "Compatible Match"
#             elif base_status == "Compatible Match":
#                 base_status = "Partial Match (Proximate)"
            
#             desc += " Temporal context differs."

#         return base_status, desc

#     def satisfy(self, ctx: CandidateContext):
#         src_code_str = ctx.src.get("context_labels")
#         tgt_code_str = ctx.tgt.get("context_labels")

#         # 1. If no composite context on either side, this constraint passes (doesn't trigger)
#         if not src_code_str and not tgt_code_str:
#             return True, None, None

#         # 2. Extract and Clean Text
#         # Assuming codes are pipe-separated like "LOINC:123|systolic|sitting"
#         # We strip the first ID often and join the rest, or just join all.
#         def clean_composite(s): 
#             if not s: return ""
#             return " ".join(s.replace("|", " ").split())
        
#         text_a = clean_composite(src_code_str)
#         text_b = clean_composite(tgt_code_str)

#         # 3. Compute Similarity
   
#         if not text_a or not text_b:
#             # One side has context, the other doesn't -> Mismatch
#             return True, "Partial Match (Tentative)", {
#                 "description": "One variable lacks the composite context present in the other."
#             }

#         # vec_a = self.embed_fn(text_a)
#         # vec_b = self.embed_fn(text_b)
#         score = similarity_score(settings.EMBEDDING_MODEL_NAME, text_a, text_b)
        
#         # 4. Determine Status
#         status, desc_suffix = self._get_status_from_score(score, ctx.visit_match)

#         details = {
#             "description": f"Composite Variable Analysis: {desc_suffix}",
#             "semantic_similarity": float(score),
#             "source_context": src_code_str,
#             "target_context": tgt_code_str
#         }

#         # This constraint is definitive for composite variables. 
#         # We return the status to stop the solver here.
#         return True, status, details
class MismatchFallbackConstraint(Constraint):
    def satisfy(self, ctx: CandidateContext):
        if ctx.src_type != ctx.tgt_type:
            return True, "Not Applicable", {"description": f"Statistical type mismatch: {ctx.src_type} vs {ctx.tgt_type}"}
        return True, None, None

# --- Solver Engine ---

class ConstraintSolver:
    def __init__(self):
        # The Pipeline of Rules
        self.constraints: List[Constraint] = [
            ContextualSimilarityConstraint(), # First check for composite variable similarity
            VisitConstraint(), # then ensure visit alignment and data type
            DataTypeConstraint(),
            UnitLogicConstraint(), # soft checks for continuous variables
            ValueSetLogicConstraint(), # soft checks for categorical variables
           # MismatchFallbackConstraint() # final fallback for mismatched types
        ]

    @staticmethod
    @lru_cache(maxsize=None)
    def check_visit_string(visit_str_src: str, visit_str_tgt:str) -> str:
        # Normalize logic
        s_low = visit_str_src.lower()
        t_low = visit_str_tgt.lower()
        for hint in settings.DATE_HINTS:
            if hint in s_low: return visit_str_tgt
            if hint in t_low: return visit_str_src
        return visit_str_src

    def solve(self, src_info: Dict, tgt_info: Dict) -> Tuple[Dict, str]:
        ctx = CandidateContext(src=src_info, tgt=tgt_info)
        
        
        for constraint in self.constraints:
            continue_check, status, details = constraint.satisfy(ctx)
            
            if status is not None:
                return details, status
            
            if not continue_check:
                # If a hard constraint fails and returns no status, we default fail
                return details or {"description": "Constraint check failed."}, "Not Applicable"

        return {"description": "No specific rule matched."}, "Partial Match (Tentative)"