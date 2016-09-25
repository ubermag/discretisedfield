# test for Issue #1 (https://github.com/joommf/discretisedfield/issues/1)

import subprocess
import pytest

@pytest.mark.xfail
def test_matplotlib_warning():
    command = """python -c "import discretisedfield" """
    status, output = subprocess.getstatusoutput(command)

    print("output = {}".format(output))
    assert status == 0
    assert len(output) == 0   # expect no warnings
