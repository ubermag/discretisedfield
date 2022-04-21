def pytest_collectstart(collector):
    if collector.fspath and collector.fspath.ext == ".ipynb":
        collector.skip_compare += (
            "application/javascript",
            "application/vnd.holoviews_load.v0+json",
        )
