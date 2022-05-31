import discretisedfield as df


def test_defaults():
    # default settings
    assert df.plotting.defaults.norm_filter
    assert df.plotting.Hv._norm_filter
    assert (
        repr(df.plotting.defaults.norm_filter)
        == "plotting defaults\n  norm_filter: True"
    )

    # disable norm filtering
    df.plotting.defaults.norm_filter = False
    assert not df.plotting.Hv._norm_filter

    # enable norm filtering
    df.plotting.defaults.norm_filter = True
    assert df.plotting.Hv._norm_filter
