class Defaults:
    """Default settings for plotting."""

    _all = ["norm_filter"]

    def __init__(self, *classes):
        self._classes = classes
        self.reset()

    def reset(self):
        """Reset values to their defaults."""
        self.norm_filter = True

    def __repr__(self):
        summary = "plotting defaults\n"
        for setting in self._all:
            summary += f"  {setting}: {getattr(self, setting)}\n"
        return summary

    def __getattr__(self, attr):
        if attr not in self._all:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {attr!r}."
            )
        return getattr(self, f"_{attr}")

    def __setattr__(self, attr, value):
        if attr in self._all:
            attr = f"_{attr}"
            for class_ in self._classes:
                setattr(class_, attr, value)
        super().__setattr__(attr, value)

    def __dir__(self):
        return dir(self.__class__) + self._all
