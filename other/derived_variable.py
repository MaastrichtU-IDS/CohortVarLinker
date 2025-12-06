"""Utility functions for adding derived variables to a cohort graph."""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

from rdflib import Graph, Literal, RDF, RDFS, URIRef
from rdflib.namespace import XSD

from .ontology_model import Concept
from .utils import OntologyNamespaces





@dataclass
class DerivedVariable:
    """Specification for a variable derived from other OMOP variables."""

    name: str
    omop_id: int
    code: str
    label: str
    unit: Optional[str] = None
    required_omops: Set[int] = field(default_factory=set)
    formula: Optional[str] = None


def find_var_by_omop_id(
    g: Graph,
    graph_uri: URIRef,
    omop_id: int,
    ns: OntologyNamespaces,
) -> Optional[URIRef]:
    """Return the variable URI for a given OMOP ID if present in the graph."""
    for s, _, _ in g.triples(
        (None, ns.CMEO.value.has_omop_id, Literal(omop_id, datatype=XSD.integer)),
        context=graph_uri,
    ):
        return s
    return None


def read_derived_variables(
    file_path: str,
) -> Iterable[DerivedVariable]:
    """Read derived variables from a json file """
    
    import json
     
    with open(file_path, 'r') as f:
        data = json.load(f)

    for item in data['DERIVED_VARIABLES']:
        yield DerivedVariable(
            name=item['name'],
            omop_id=item['omop_id'],
            code=item['code'],
            label=item['label'],
            unit=item.get('unit'),
            required_omops=set(item.get('required_omops', [])),
            formula=item.get('formula')
        )
    
   
def add_derived_variables(
    g: Graph,
    cohort_id: str,
    cohort_graph: URIRef,
    derived_vars: Iterable[DerivedVariable],
    ns: OntologyNamespaces,
    find_var_by_omop_id_func,
    add_standardization_func,
) -> Graph:
    """Add derived variables and their derivation processes to the graph."""

    for dv in derived_vars:
        out_var_uri = find_var_by_omop_id_func(g, cohort_graph, dv.omop_id, ns)
        if out_var_uri is None:
            out_var_uri = URIRef(f"{ns.CMEO.value}{cohort_id}/{dv.name.replace(' ', '_').lower()}")
            g.add((out_var_uri, RDF.type, ns.CMEO.value.data_element, cohort_graph))
            concept = [
                Concept(standard_label=dv.label, code=dv.code, omop_id=dv.omop_id)
            ]
            g = add_standardization_func(g, out_var_uri, concept, cohort_graph)
            g.add((out_var_uri, RDFS.label, Literal(dv.label), cohort_graph))
            g.add((out_var_uri, ns.CMEO.value.has_omop_id, Literal(dv.omop_id, datatype=XSD.integer), cohort_graph))
            g.add((out_var_uri, ns.CMEO.value.has_value, Literal(dv.code, datatype=XSD.string), cohort_graph))

        input_vars: List[URIRef] = []
        missing = []
        for omop in dv.required_omops:
            uri = find_var_by_omop_id_func(g, cohort_graph, omop, ns)
            if uri is not None:
                input_vars.append(uri)
            else:
                missing.append(omop)
        if missing:
            print(f"[{dv.name}] Skipping: required OMOP(s) not found: {missing}")
            continue

        process_uri = URIRef(f"{out_var_uri}/derivation_process")
        g.add((process_uri, RDF.type, ns.CMEO.value.data_transformation, cohort_graph))
        g.add((process_uri, RDFS.label, Literal(f"{dv.label} calculation process"), cohort_graph))
        formula_str = dv.formula or f"Derived using OMOPs: {', '.join(map(str, dv.required_omops))}"
        g.add((process_uri, ns.CMEO.value.formula, Literal(formula_str, datatype=XSD.string), cohort_graph))

        for inp in input_vars:
            g.add((process_uri, ns.OBI.value.has_specified_input, inp, cohort_graph))
            add_standardization_func(g, inp, [], cohort_graph)

        g.add((process_uri, ns.OBI.value.has_specified_output, out_var_uri, cohort_graph))
        g.add((out_var_uri, ns.OBI.value.is_specified_output_of, process_uri, cohort_graph))

        print(f"[{dv.name}] Added derivation process for: {out_var_uri} using {input_vars}")

    return g