import pytest

def test_fx(pat_text_with_numbered_parts):
    assert '250' in pat_text_with_numbered_parts

@pytest.mark.skip(reason='Not yet implemented')
def test_clean_np():
    assert False

@pytest.mark.skip(reason='Not yet implemented')
def test_set_config():
    assert False

@pytest.mark.skip(reason='Not yet implemented')
def test_get_nps():
    assert False

@pytest.mark.skip(reason='Not yet implemented')
def test_prep_report():
    assert False

@pytest.mark.skip(reason='Not yet implemented')
def test_report():
    assert False

@pytest.mark.skip(reason='Not yet implemented')
def test_report_json():
    assert False
