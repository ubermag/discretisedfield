import discretisedfield as df


def test_version():
    assert isinstance(df.__version__, str)
    assert '.' in df.__version__


def test_dependencies():
    assert isinstance(df.__dependencies__, list)
    assert len(df.__dependencies__) > 0
