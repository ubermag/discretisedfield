# test for Issue #1 (https://github.com/joommf/discretisedfield/issues/1)

# This test may catch any warnings/print statements that are
# unintentionally issued during import.

import subprocess
import pytest




def test_matplotlib_warning_setup():
    # On travis, matplotlib builds the font cache the first time
    # we import it. Thus, need to import before we do the actual
    # test
    # "UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment."

    command = """python -c "import matplotlib.pyplot" """
    status, output = subprocess.getstatusoutput(command)
    print("output = {}".format(output))
    assert status == 0
    if len(output) > 0:
        assert  'building the font cache' in output
    pass
    

def test_matplotlib_warning():
    
    command = """python -c "import discretisedfield" """
    status, output = subprocess.getstatusoutput(command)

    print("output = {}".format(output))
    assert status == 0
    if len(output) == 0:
        pass  # all good

    # On travis, matplotlib builds the font cache the first time
    # we import it. Thus, we may get this warning here, which is not an error.
    # "UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment."
    elif "UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment." in output and 500 < len(output) < 600:
        pass
    else:
        # raise error
        assert len(output) == 0   # expect no warnings 
