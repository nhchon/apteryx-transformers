import pytest

def test_fx(pat_text_with_numbered_parts):
    assert '250' in pat_text_with_numbered_parts

@pytest.mark.skip(reason='Not yet implemented')
def test_clean_np():
    assert False

@pytest.mark.skip(reason='Not yet implemented')
def test_set_config():
    assert False


def test_get_nps(small_np, pat_text_no_numbered):
    print(small_np.get_nps(pat_text_no_numbered))


@pytest.mark.skip(reason='Not yet implemented')
def test_prep_report():
    assert False


def test_report_w_nums(small_np, pat_text_with_numbered_parts):
    report = small_np.report(pat_text_with_numbered_parts)
    assert 'main' in report

def test_report_no_nums(small_np, pat_text_no_numbered):
    report = small_np.report(pat_text_no_numbered)
    assert 'main' in report


@pytest.mark.skip(reason='Not yet implemented')
def test_report_json():
    assert False
