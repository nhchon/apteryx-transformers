import pytest

@pytest.fixture
def pat_text_with_numbered_parts() -> str:
    return 'The detection sensor 250 may deliver the acquired location information of the pedestrian to the controller 100. Also, the detection sensor 250 may acquire the coordinate information that varies with the movement of the pedestrian and the distance between the vehicle 1 and the pedestrian and then deliver the coordinate information and the distance to the controller 100.'

@pytest.fixture
def pat_text_no_numbered() -> str:
    return 'The present disclosure is concerned with a visual apparatus and a method for creation of artificial vision. In particular, the present disclosure provides an interface and method for controlling a visual prosthesis (i.e. device) implanted in an individual patient (i.e. subject) to create artificial vision. FIG. 1 shows a visual prosthesis apparatus'