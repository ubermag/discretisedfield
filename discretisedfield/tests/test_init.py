import discretisedfield as df


def test_version():
    assert isinstance(df.__version__, str)
    assert "." in df.__version__
