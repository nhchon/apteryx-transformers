import pytest

from parsers.noun_phrase_parser import NPParser


@pytest.fixture(scope="session")
def small_np() -> NPParser:
    return NPParser(spacy_model='en_core_web_sm')

@pytest.fixture
def pat_text_with_numbered_parts() -> str:
    return 'The detection sensor 250 may deliver the acquired location information of the pedestrian to the controller 100. Also, the detection sensor 250 may acquire the coordinate information that varies with the movement of the pedestrian and the distance between the vehicle 1 and the pedestrian and then deliver the coordinate information and the distance to the controller 100.'

@pytest.fixture
def pat_text_no_numbered() -> str:
    return 'The present disclosure is concerned with a visual apparatus and a method for creation of artificial vision. In particular, the present disclosure provides an interface and method for controlling a visual prosthesis (i.e. device) implanted in an individual patient (i.e. subject) to create artificial vision. FIG. 1 shows a visual prosthesis apparatus'

@pytest.fixture
def table_text() -> str:
    return 'The data are presented in the following table. \n     \n       \n         \n               \n             \n                   \n               \n                 Rotary flexion test results \n               \n                 Steel AISI/SAE 1045 \n               \n               \n               \n               \n               \n               \n             \n                   \n                 Test \n                 S y   \n                 Stress \n                 Cycles to \n               \n                   \n                 tube No \n                 Percentage \n                 (MPa) \n                 failure \n               \n                   \n                   \n               \n               \n               \n               \n               \n               \n             \n                   \n                 1 \n                 75% \n                 573 \n                 27328 \n               \n                   \n                 2 \n                 75% \n                 573 \n                 '

@pytest.fixture
def table_text_parsed_severity_1() -> str:
    return 'The data are presented in the following table. '