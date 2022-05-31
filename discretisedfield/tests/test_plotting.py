import discretisedfield as df


def test_defaults():
    # default settings
    assert df.plotting.defaults.norm_filter
    assert df.plotting.Hv._norm_filter

    # disable norm filtering
    df.plotting.defaults.norm_filter = False
    assert not df.plotting.Hv._norm_filter

    # enable norm filtering
    df.plotting.defaults.norm_filter = True
    assert df.plotting.Hv._norm_filter
