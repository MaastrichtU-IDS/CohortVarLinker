from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd
# from SPARQLWrapper import JSON, SPARQLWrapper

from collections import defaultdict
from .config import settings
from .utils import apply_rules, execute_query
from .vector_db import search_in_db
from .modes import MappingType, EmbeddingType 
import json
from dataclasses import dataclass
from typing import FrozenSet, Optional

BASELINE_TIME_HINTS = ["6 months prior to baseline", "prior to baseline visit"]
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

def check_visit_string(visit_str_src: str, visit_str_tgt:str) -> str:
    # if src or tgt visit string contains any of the time hints, return the value of the visit that is not in time hint
    # print(f"Checking visit strings: src='{visit_str_src}', tgt='{visit_str_tgt}'")
    
    for hint in DATE_HINTS:
        if hint in visit_str_src.lower():
            return visit_str_tgt
        if hint in visit_str_tgt.lower():
            return visit_str_src

    for hint in BASELINE_TIME_HINTS:
        if hint in visit_str_src.lower() or hint in visit_str_tgt.lower():
            if 'follow-up' in visit_str_src.lower() or 'follow-up' in visit_str_tgt.lower():
                return visit_str_src
            return 'baseline time'
    return visit_str_src


def fetch_study_graph(study_name:str, graph_repo:str) -> str:
    return f"""
        PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dc:    <http://purl.org/dc/elements/1.1/>
        PREFIX obi:   <http://purl.obolibrary.org/obo/obi.owl/>
        PREFIX iao:   <http://purl.obolibrary.org/obo/iao.owl/>
        PREFIX cmeo:  <https://w3id.org/CMEO/>

        SELECT
        ("{study_name}" AS ?study_name)
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
                (GROUP_CONCAT(DISTINCT ?cv;  SEPARATOR="|") AS ?code_value)
                (GROUP_CONCAT(DISTINCT ?cl;  SEPARATOR="|") AS ?code_label)
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


def _split_pipe(x: Optional[str]) -> FrozenSet[str]:
    if not x or pd.isna(x):
        return frozenset()
    return frozenset([t.strip() for t in str(x).split("|") if t.strip()])

@dataclass(frozen=True)
class VariableProfile:
    study: str
    var_name: str                  # dc:identifier
    var_uri: Optional[str]         # add this to SELECT if possible
    category: Optional[str]
    stat_type: Optional[str]
    data_type: Optional[str]
    unit: Optional[str]
    visit_labels: FrozenSet[str]
    code_values: FrozenSet[str]
    code_labels: FrozenSet[str]
    omop_ids: FrozenSet[str]
    cat_labels: FrozenSet[str]
    cat_values: FrozenSet[str]

def build_profiles(df: pd.DataFrame) -> list[VariableProfile]:
    # choose a stable key: var_uri preferred; else var_name
    key = "dataElementA" if "dataElementA" in df.columns else "var_nameA"

    profiles = []
    for _, g in df.groupby(key, dropna=False):
        study = g["study_name"].iloc[0] if "study_name" in g.columns else None
        var_name = g["var_nameA"].dropna().astype(str).iloc[0]

        prof = VariableProfile(
            study=study,
            var_name=var_name,
            var_uri=g["dataElementA"].dropna().astype(str).iloc[0] if "dataElementA" in g.columns and g["dataElementA"].notna().any() else None,
            category=g["category"].dropna().astype(str).iloc[0] if "category" in g.columns and g["category"].notna().any() else None,
            stat_type=g["stat_type_value"].dropna().astype(str).iloc[0] if "stat_type_value" in g.columns and g["stat_type_value"].notna().any() else None,
            data_type=g["data_type_value"].dropna().astype(str).iloc[0] if "data_type_value" in g.columns and g["data_type_value"].notna().any() else None,
            unit=g["unit_label"].dropna().astype(str).iloc[0] if "unit_label" in g.columns and g["unit_label"].notna().any() else None,
            visit_labels=frozenset(g["visit"].dropna().astype(str).unique()) if "visit" in g.columns else frozenset(),
            code_values=_split_pipe(g["code_value"].dropna().astype(str).iloc[0]) if "code_value" in g.columns and g["code_value"].notna().any() else frozenset(),
            code_labels=_split_pipe(g["code_label"].dropna().astype(str).iloc[0]) if "code_label" in g.columns and g["code_label"].notna().any() else frozenset(),
            omop_ids=_split_pipe(g["omop_id"].dropna().astype(str).iloc[0]) if "omop_id" in g.columns and g["omop_id"].notna().any() else frozenset(),
            cat_labels=_split_pipe(g["categoryies_label"].dropna().astype(str).iloc[0]) if "categoryies_label" in g.columns and g["categoryies_label"].notna().any() else frozenset(),
            cat_values=_split_pipe(g["categoryies_value"].dropna().astype(str).iloc[0]) if "categoryies_value" in g.columns and g["categoryies_value"].notna().any() else frozenset(),
        )
        profiles.append(prof)

    return profiles
