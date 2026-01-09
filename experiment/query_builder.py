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
        # Optimized query structure
        return f""" 
            {cls.PREFIXES}
            SELECT ?omop_id
            (SAMPLE(?lbl) AS ?code_label)
            (SAMPLE(?val) AS ?code_value)
            (GROUP_CONCAT(DISTINCT ?src_combined; SEPARATOR=" | ") AS ?source_data)
            (GROUP_CONCAT(DISTINCT ?tgt_combined; SEPARATOR=" | ") AS ?target_data)
            (GROUP_CONCAT(DISTINCT ?src_val; SEPARATOR="|") AS ?source_domain)
            (GROUP_CONCAT(DISTINCT ?tgt_val; SEPARATOR="|") AS ?target_domain)
            WHERE {{
                {{
                    SELECT ?omop_id ?lbl ?val ?src_combined ?src_val
                    WHERE {{
                        GRAPH <{graph_repo}/{source}> {{
                             ?deA dc:identifier ?src_var ;
                                  skos:has_close_match ?codeSetA .
                             ?codeSetA rdf:_1 ?codeNodeA .
                             ?codeNodeA rdfs:label ?lbl ;
                                        cmeo:has_value ?val ;
                                        iao:denotes/cmeo:has_value ?omop_id .
                             OPTIONAL {{ ?deA sio:has_attribute/skos:has_close_match/rdfs:label ?src_vis_label . }}
                             OPTIONAL {{ ?deA sio:has_annotation/cmeo:has_value ?src_val . }}
                             BIND(IF(BOUND(?src_vis_label), CONCAT(?src_var, "[", ?src_vis_label, "]"), ?src_var) AS ?src_combined)
                        }}
                    }}
                }}
                UNION
                {{
                    # ... Target Study Block (Logic Preserved) ...
                     SELECT ?omop_id ?lbl ?val ?tgt_combined ?tgt_val
                    WHERE {{
                        GRAPH <{graph_repo}/{target}> {{
                             ?deB dc:identifier ?tgt_var ;
                                  skos:has_close_match ?codeSetB .
                             ?codeSetB rdf:_1 ?codeNodeB .
                             ?codeNodeB rdfs:label ?lbl ;
                                        cmeo:has_value ?val ;
                                        iao:denotes/cmeo:has_value ?omop_id .
                             OPTIONAL {{ ?deB sio:has_attribute/skos:has_close_match/rdfs:label ?tgt_vis_label . }}
                             OPTIONAL {{ ?deB sio:has_annotation/cmeo:has_value ?tgt_val . }}
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
             SELECT ?identifier
             ?stat_label
             ?unit_label
             ?data_type_val
             (COALESCE(?cat_labels_str, "") AS ?all_cat_labels)
             (COALESCE(?cat_values_str, "") AS ?all_original_cat_values)
             (COALESCE(?code_labels_str, "") AS ?code_label)
             (COALESCE(?code_values_str, "") AS ?code_value)
            WHERE {{
                GRAPH <{graph_repo}/{source}> {{
                   VALUES ?identifier {{ {values_str} }}
                   ?dataElement a cmeo:data_element ; dc:identifier ?identifier .
                   
                    # 2. STATS & DATA TYPE (Simple 1:1 lookups)
                    
                   OPTIONAL {{ ?dataElement iao:is_denoted_by/cmeo:has_value ?stat_label . }}
                   OPTIONAL {{
                        {{ ?data_type sio:is_attribute_of ?dataElement . }} UNION {{ ?dataElement sio:has_attribute ?data_type . }}
                        ?data_type a cmeo:data_type ; cmeo:has_value ?data_type_val .
                   }}
                    # 3. UNIT (1:1 lookup)
                    OPTIONAL {{
                        ?dataElement obi:has_measurement_unit_label/skos:has_close_match/cmeo:has_value ?unit_label .
                    }}

                    # 4. CODES (Subquery to prevent fan-out)
                   OPTIONAL {{
                    SELECT ?dataElement 
                            (GROUP_CONCAT(DISTINCT ?cL; SEPARATOR="|") AS ?code_labels_str)
                            (GROUP_CONCAT(DISTINCT ?cV; SEPARATOR="|") AS ?code_values_str)
                    WHERE {{
                            VALUES ?identifier {{ {values_str} }}
                            ?dataElement a cmeo:data_element;
                                dc:identifier ?identifier .
                            ?dataElement skos:has_close_match ?codeSetA. 

                            # We rely on the object type (cmeo:code) to filter relevant links.
                            ?codeSetA ?p ?codeNodeA .  
                            ?codeNodeA a cmeo:code ;
                                    rdfs:label ?cL ;
                                    cmeo:has_value ?cV .
                    }} GROUP BY ?dataElement
                    }}
                    
                        # 5. CATEGORIES (Subquery to prevent fan-out)
                    # -------------------------------
                    OPTIONAL {{
                    SELECT ?dataElement 
                            (GROUP_CONCAT(DISTINCT ?catL; SEPARATOR="; ") AS ?cat_labels_str)
                            (GROUP_CONCAT(DISTINCT ?catV; SEPARATOR="; ") AS ?cat_values_str)
                    WHERE {{
                        VALUES ?identifier {{ {values_str} }}
                        ?dataElement a cmeo:data_element;
                            dc:identifier ?identifier .
                        ?cat_val a obi:categorical_value_specification ;
                                obi:specifies_value_of ?dataElement ;
                                cmeo:has_value ?catV ;
                                skos:has_close_match/rdfs:label ?catL .
                    }} GROUP BY ?dataElement
                    }}
                                
                }}
            }}
        """