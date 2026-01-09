import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from .query_builder import SPARQLQueryBuilder
from .utils import execute_query

class VariableProfile:
    @staticmethod
    def _fetch_chunk(chunk: list, study_name: str, graph_repo: str) -> list:
        results = []
        try:
            values_str = " ".join(f'"{v}"' for v in chunk)
            query = SPARQLQueryBuilder.build_statistic_query(study_name, values_str, graph_repo)
            
            response = execute_query(query)
            bindings = response.get("results", {}).get("bindings", [])
            
            for res in bindings:
                identifier = res['identifier']['value']
                # Helper to safely extract string
                def val(k): return res[k]['value'] if k in res and res[k]['value'].strip() else None
                def joined_sep(k):
                    return val(k).split("|") if val(k) else []
                # the composite code
                results.append({
                    'identifier': identifier,
                    'stat_label': val('stat_label'),
                    'unit_label': val('unit_label'),
                    'data_type': val('data_type_val'),
                    "categories_labels": val('all_cat_labels'),
                    'original_categories': val('all_original_cat_values'),
                    'composite_code_labels': joined_sep('code_label'),
                    'composite_code_values': joined_sep('code_value')
                })
        except Exception as e:
            print(f"Error chunk fetch: {e}")
        return results

    @classmethod
    def attach_attributes(cls, df: pd.DataFrame, src_study: str, tgt_study: str, graph_repo: str) -> pd.DataFrame:
        src_vars = df["source"].dropna().unique().tolist()
        tgt_vars = df["target"].dropna().unique().tolist()
        
        def run_fetch(vars_, study):
            data = []
            if not vars_: return pd.DataFrame()
            chunks = [vars_[i:i+50] for i in range(0, len(vars_), 50)]
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(cls._fetch_chunk, c, study, graph_repo): c for c in chunks}
                for f in as_completed(futures):
                    data.extend(f.result())
            return pd.DataFrame(data)

        src_df = run_fetch(src_vars, src_study)
        tgt_df = run_fetch(tgt_vars, tgt_study)

        if not src_df.empty:
            df = df.merge(src_df.rename(columns={
                "identifier": "source", "stat_label": "source_type", 
                "unit_label": "source_unit", "data_type": "source_data_type",
                "original_categories": "source_original_categories", "composite_code_labels": "source_composite_code_labels",
            }), on="source", how="left")
            
        if not tgt_df.empty:
            df = df.merge(tgt_df.rename(columns={
                "identifier": "target", "stat_label": "target_type",
                "unit_label": "target_unit", "data_type": "target_data_type",
                "original_categories": "target_original_categories",
                "composite_code_labels": "target_composite_code_labels",
            }), on="target", how="left")
            
        return df