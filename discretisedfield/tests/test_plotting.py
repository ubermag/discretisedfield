import pytest

import discretisedfield as df


def test_defaults():
    # default settings
    assert df.plotting.defaults.norm_filter
    assert df.plotting.Hv._norm_filter
    assert repr(df.plotting.defaults) == "plotting defaults\n  norm_filter: True\n"

    assert list(df.plotting.defaults) == ["norm_filter"]
    assert all(key in dir(df.plotting.defaults) for key in df.plotting.defaults)

    with pytest.raises(AttributeError):
        df.plotting.defaults.nonexisting

    # disable norm filtering
    df.plotting.defaults.norm_filter = False
    assert not df.plotting.defaults.norm_filter
    assert not df.plotting.Hv._norm_filter

    # enable norm filtering
    df.plotting.defaults.reset()
    assert df.plotting.defaults.norm_filter
    assert df.plotting.Hv._norm_filter
