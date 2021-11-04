
import pytest


'''     defining fixture functions   '''

# 1st fixture function to give image path with label
@pytest.fixture
def input_value():
    file = 'Complex_dataSet/IMG/img_2021_02_10-05_38_34_25.jpg'
    label = 2.5
    return file, label
