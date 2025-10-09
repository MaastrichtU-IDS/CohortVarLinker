import pytest
from src import utils

def test_normalize_text_basic():
    assert utils.normalize_text(" Hello World ") == "hello_world"

def test_normalize_text_none():
    assert utils.normalize_text(None) is None
    assert utils.normalize_text("nan") is None

def test_extract_age_range_between():
    assert utils.extract_age_range("between 10 and 20 years") == (10.0, 20.0)

def test_extract_age_range_operators():
    assert utils.extract_age_range("age >= 18 years and <= 65 years") == (18.0, 65.0)

def test_get_cohort_and_var_uri():
    cohort_uri = utils.get_cohort_uri("Study A")
    var_uri = utils.get_var_uri("Study A", "Blood Pressure")
    base = str(utils.OntologyNamespaces.CMEO.value)
    assert str(cohort_uri) == base + "study_a"
    assert str(var_uri) == base + "study_a/data_element/blood_pressure"
    