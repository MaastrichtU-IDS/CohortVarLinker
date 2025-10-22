#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install pandas requests

import sys
import pandas as pd
import requests
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

ATHENA_ENDPOINT = "https://athena.ohdsi.org/api/v1/concepts"
TIMEOUT = 25

# ---------------------- helpers ----------------------

def split_code_prefixed(val: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """'loinc:8302-2' -> ('LOINC','8302-2')"""
    if not val or ":" not in str(val):
        return (None, None)
    pre, post = str(val).split(":", 1)
    vocab = pre.strip().upper()
    if vocab in {"SNOMEDCT", "SNOMED-CT"}:
        vocab = "SNOMED"
    elif vocab == "OMOP":
        vocab = "OMOP Extension"
    elif vocab == "RXNORM":
        vocab = "RxNorm"
    return vocab, post.strip()

def to_int_or_none(x: Optional[str]) -> Optional[int]:
    if x is None or str(x).strip() == "" or str(x).strip().lower() == "nan":
        return None
    try:
        return int(str(x).strip())
    except Exception:
        return None

def athena_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://athena.ohdsi.org/",
    }

def fetch_athena(label: str, vocabulary: str, include_classification: bool = True, page_size: int = 50) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "vocabulary": vocabulary,
       
        "invalidReason": "Valid",
        "page": 1,
        "pageSize": 5,

        # "standardConcept": ["Standard"] + (["Classification"] if include_classification else []),
        "query": label,
    }
    # if domain:
    #     params["domain"] = domain
    print(f"Querying Athena for label='{label}', vocabulary='{vocabulary}'")
    print("whole url:", ATHENA_ENDPOINT + "?" + "&".join(f"{k}={v}" for k,v in params.items())  )
    r = requests.get(ATHENA_ENDPOINT, params=params, headers=athena_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "content" in data:
        return data["content"]
    if isinstance(data, list):
        return data
    return []

def rows_to_simple(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            out.append({
                "concept_id": int(r.get("id") or r.get("conceptId")),
                "name": r.get("name") or r.get("conceptName") or "",
                "code": r.get("code") or r.get("conceptCode") or "",
                "vocabulary": r.get("vocabulary") or r.get("vocabularyId") or "",
                "domain": r.get("domain") or r.get("domainId") or "",
                "standard": r.get("standardConcept"),
            })
        except Exception:
            continue
    return out

@dataclass
class ValidationLog:
    status: str                 # PASS | FAIL | N/A
    description: str
    vocabulary: Optional[str] = None
    label: Optional[str] = None
    assigned_concept_id: Optional[int] = None
    assigned_concept_code: Optional[str] = None
    matched_concept_id: Optional[int] = None
    matched_concept_code: Optional[str] = None
    matched_name: Optional[str] = None
    matched_domain: Optional[str] = None
    matched_standard: Optional[str] = None

def validate_component(*, label: str, code_prefixed: Optional[str], omop_id_str: Optional[str]) -> ValidationLog:
    lab = " ".join((label or "").strip().split())
    vocab, code = split_code_prefixed(code_prefixed)
    cid = to_int_or_none(omop_id_str)
    if vocab and vocab == "ICARE":
        return ValidationLog(status="PASS", description="its custom vocabulary.")
    if not vocab and not code and cid is None:
        return ValidationLog(status="N/A", description="No code/ID provided for this component.")
    if not lab:
        return ValidationLog(status="FAIL", description="Missing label; cannot query Athena.", vocabulary=vocab)
    if not vocab:
        return ValidationLog(status="FAIL", description="Vocabulary prefix missing/unknown in code.", label=lab)

    rows = fetch_athena(code, vocab, include_classification=True)
    concepts = rows_to_simple(rows)
    if not concepts:
        return ValidationLog(status="FAIL", description="Athena returned no candidates.", vocabulary=vocab, label=lab)

    id_hits = {c["concept_id"]: c for c in concepts}
    code_hits = {c["code"]: c for c in concepts}

    if cid is not None and code:
        c1, c2 = id_hits.get(cid), code_hits.get(code)
        if c1 and c2 and c1["concept_id"] == c2["concept_id"]:
            c = c1
            return ValidationLog("PASS", "Exact match on concept_id and concept_code.", vocab, lab, cid, code,
                                 c["concept_id"], c["code"], c["name"], c["domain"], c["standard"])
        if c1 and not c2:
            return ValidationLog("FAIL", "concept_id matched but concept_code not found for this label/vocabulary.",
                                 vocab, lab, cid, code, c1["concept_id"], None, c1["name"], c1["domain"], c1["standard"])
        if c2 and not c1:
            return ValidationLog("FAIL", "concept_code matched but concept_id not found for this label/vocabulary.",
                                 vocab, lab, cid, code, None, c2["code"], c2["name"], c2["domain"], c2["standard"])
        return ValidationLog("FAIL", "Neither concept_id nor concept_code matched Athena candidates.", vocab, lab, cid, code)

    if cid is not None:
        c = id_hits.get(cid)
        if c:
            return ValidationLog("PASS", "Exact match on concept_id.", vocab, lab, cid, None,
                                 c["concept_id"], c["code"], c["name"], c["domain"], c["standard"])
        return ValidationLog("FAIL", "Assigned concept_id not found in Athena candidates.", vocab, lab, cid, None)

    if code:
        c = code_hits.get(code)
        if c:
            return ValidationLog("PASS", "Exact match on concept_code.", vocab, lab, None, code,
                                 c["concept_id"], c["code"], c["name"], c["domain"], c["standard"])
        return ValidationLog("FAIL", "Assigned concept_code not found in Athena candidates.", vocab, lab, None, code)

    return ValidationLog("N/A", "No code/id provided.")

def overall_status(*components: ValidationLog) -> str:
    present = [c.status for c in components if c.status != "N/A"]
    if not present:
        return "N/A"
    return "FAIL" if any(s == "FAIL" for s in present) else "PASS"

# Optional mapper: if your CSV puts OMOP **table** names in domain, normalize to OMOP **Domain** strings
# DOMAIN_MAP = {
#     "measurement": "Measurement",
#     "condition_occurrence": "Condition",
#     "observation": "Observation",
#     "procedure_occurrence": "Procedure",
#     "person": "Person",
#     "observation_period": "Observation Period",
#     # add more if needed
# }

# ---------------------- main ----------------------

        
def run(in_csv: str, out_csv: str) -> None:
    df = pd.read_csv(in_csv, low_memory=False)
    df = df.applymap(lambda x: None if pd.isna(x) or (isinstance(x, str) and x.strip() == "") else x)

    print(df.head(2))
    # lower-case all column names
    df.columns = [c.lower() for c in df.columns]
    # convert all values to lower-case for required columns
    # expected columns (lowercase)
    need = [
        "variablename","variablelabel","domain", "categorical", "units","visits",
        "categorical value concept name","categorical value concept code","categorical value omop id",
        "variable concept name","variable concept code","variable omop id",
        "additional context concept name","additional context concept code","additional context omop id",
        "unit concept name","unit concept code","unit omop id",
        "visit concept name","visit concept code","visit omop id",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
   
    out_rows = []
    for _, r in df.iterrows():
        varlabel = r.get("variablelabel", "")
        # normalize domain (optional, safe to pass raw)
        raw_domain = r.get("domain", "").strip()
        categories = r.get("categorical","")
        unit = r.get("units","")
        visit = r.get("visits","")
         
        print(f"Validating variable '{r.get('variablename','')}' with domain '{raw_domain}'")
        # variable
        if varlabel != "":
            if r.get("variable concept name",None) is None and r.get("variable concept code",None) is None and r.get("variable omop id",None) is None:
                v_log = ValidationLog(status="N/A", description="Variable concept details missing for non-empty variable label.")
            else:
                v_log = validate_component(
                    label=r.get("variable concept name", "") or varlabel,
                    code_prefixed=r.get("variable concept code", ""),
                    omop_id_str=r.get("variable omop id", ""),
                    # domain_hint=domain,
                )
        print(f"Validated variable '{r.get('variablename','')}' - Status: {v_log.status} - Reason: {v_log.description}")
        # categorical values
        if categories != "":
            if r.get("categorical value concept name",None) is None and r.get("categorical value concept code",None) is None and r.get("categorical value omop id",None) is None:
                cv_logs = ValidationLog(status="N/A", description="Categorical value concept details missing for categorical variable.")
            else:
                # split the categories and check each one and its corresponding code and id existance
                categories_list = categories.split("|")
                cat_name_list = str(r.get("categorical value concept name","")).split("|")
                cat_code_list = str(r.get("categorical value concept code","")).split("|")
                cat_id_list = str(r.get("categorical value omop id","")).split("|")
                # check for mismatch and print error
                if not len(categories_list) == len(cat_name_list):
                    cv_logs = ValidationLog(status="FAIL", description=f"Mismatch in number {len(categories_list)} of categorical values and their {len(cat_name_list)} concept details.")
                elif not len(cat_name_list) == len(cat_code_list) == len(cat_id_list):
                    cv_logs = ValidationLog(status="FAIL", description=f"Mismatch in number of categorical value concept details: names({len(cat_name_list)}), codes({len(cat_code_list)}), ids({len(cat_id_list)}).")
                else:
                    
                    cv_log_list = []    
                    for idx in range(len(cat_name_list)):

                        cv_log = validate_component(
                            label=cat_name_list[idx] if idx < len(cat_name_list) else "",
                            code_prefixed=cat_code_list[idx] if idx < len(cat_code_list) else "",
                            omop_id_str=cat_id_list[idx] if idx < len(cat_id_list) else "",
                            # domain_hint=domain,
                        )
                    
                        cv_log_list.append(cv_log)
                    cv_logs_str = "; ".join([f"{cv_log.label}= {cv_log.status}: {cv_log.description} " for cv_log in cv_log_list])
                    cv_logs = ValidationLog(status="PASS" if all(cv_log.status=="PASS" for cv_log in cv_log_list) else "FAIL", description=cv_logs_str)
                    print(f"Validated categorical values for variable '{r.get('variablename','')}' - Status: {cv_logs_str}")
        else:
                    cv_logs = ValidationLog(status="N/A", description="No categorical values provided.")
        # additional context
        if r.get("additional context concept name",None) is not None and r.get("additional context concept code",None) is not None and r.get("additional context omop id",None) is not None:
            
            list_names = str(r.get("additional context concept name","")).split("|")
            list_codes = str(r.get("additional context concept code","")).split("|")
            list_ids = str(r.get("additional context omop id","")).split("|")
            ac_log_list = []    
            for idx in range(len(list_names)):
                
                ac_log = validate_component(
                    label=list_names[idx] if idx < len(list_names) else "",
                    code_prefixed=list_codes[idx] if idx < len(list_codes) else "",
                    omop_id_str=list_ids[idx] if idx < len(list_ids) else "",
                    # domain_hint=domain,
                )
            
                ac_log_list.append(ac_log)
            ac_logs_str = "; ".join([f"{ac_log.label}= {ac_log.status}: {ac_log.description} " for ac_log in ac_log_list])
            ac_logs = ValidationLog(status="PASS" if all(ac_log.status=="PASS" for ac_log in ac_log_list) else "FAIL", description=ac_logs_str)
            print(f"Validated additional context for variable '{r.get('variablename','')}' - Status: {ac_logs_str}")
        else:
            ac_logs = ValidationLog(status="N/A", description="No additional context provided.")
        # unit
        if unit != "":
            if r.get("unit concept name",None) is None and r.get("unit concept code",None) is None and r.get("unit omop id",None) is None:
                u_log = ValidationLog(status="N/A", description="Unit concept details missing for non-empty unit.")
            else:
                print(f"Validating unit component.. {r.get('unit concept name', '')}, {r.get('unit concept code', '')}, {r.get('unit omop id', '')}")
                u_log = validate_component(
                label=r.get("unit concept name", ""),
                code_prefixed=r.get("unit concept code", ""),
                omop_id_str=r.get("unit omop id", ""),
                # domain_hint=domain,
            )
            print(f"Validated unit for variable '{r.get('variablename','')}' - Status: {u_log.status} - Reason: {u_log.description}")
        # visit
        if visit != "":
            if r.get("visit concept name",None) is None and r.get("visit concept code",None) is None and r.get("visit omop id",None) is None:
                vi_log = ValidationLog(status="N/A", description="No visit provided.")
            else:
                vi_log = validate_component(
                    label=r.get("visit concept name", ""),
                    code_prefixed=r.get("visit concept code", ""),
                    omop_id_str=r.get("visit omop id", ""),
                    # domain_hint=domain,
                )
            print(f"Validated visit for variable '{r.get('variablename','')}' - Status: {vi_log.status} - Reason: {vi_log.description}")
        # complete_passing = all(log.status == "PASS" for log in [v_log, u_log, vi_log] if log.status != "N/A")
        out_rows.append({
            "variablename": r.get("variablename",""),
            "variablelabel": varlabel,
            "domain": raw_domain,
            "categorical_value_status": cv_logs.status,
            "categorical_value_reason": cv_logs.description,
            "variable_status": v_log.status, 
            "variable_reason": v_log.description,
            "additional_context_status": ac_logs,
            "unit_status": u_log.status, "unit_reason": u_log.description,
            "visit_status": vi_log.status, "visit_reason": vi_log.description,
            "overall_status": overall_status(cv_logs, v_log, ac_logs, u_log, vi_log),
            # 'validation_status': 'PASS' if complete_passing else 'CHECK AGAIN'
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate cde_against athena system, params: input.csv output.csv")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
