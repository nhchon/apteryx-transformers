from apteryx_transformers.parsers.parser_utils import *

def test_remove_tables(table_text, table_text_parsed_severity_1):
    parsed = remove_tables(table_text, severity=1)
    assert parsed == table_text_parsed_severity_1
