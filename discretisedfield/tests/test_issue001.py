# test for Issue #1 (https://github.com/joommf/discretisedfield/issues/1)

# This test may catch any warnings/print statements that are
# unintentionally issued during import.

import subprocess
import pytest


@pytest.mark.xfail
def test_matplotlib_warning():
    command = """python3 -c "import discretisedfield" """
    status, output = subprocess.getstatusoutput(command)

    assert status == 0
    assert len(output) == 0
