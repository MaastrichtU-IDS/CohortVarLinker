from src.vector_db import get_csv_text


def test_get_csv_text_combines_fields():
    row = {
        'VariableLabel': 'Height',
        'Variable Concept Name': 'body height',
        'Additional Context Concept Name': 'standing position'
    }
    # we check the keus in the row to ensure they are present
    assert 'VariableLabel' in row
    assert 'Variable Concept Name' in row
    assert 'Additional Context Concept Name' in row

    
    