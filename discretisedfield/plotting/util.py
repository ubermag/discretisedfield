class Defaults:
    """Default settings for plotting."""

    _defaults = {
        "norm_filter": True,
    }

    def __init__(self, *classes):
        self._classes = classes
        self.reset()

    def reset(self):
        """Reset values to their defaults."""
        for setting in self:
            setattr(self, setting, self._defaults[setting])

    def __repr__(self):
        summary = "plotting defaults\n"
        for setting in self._defaults:
            summary += f"  {setting}: {getattr(self, setting)}\n"
        return summary

    def __getattr__(self, attr):
        if attr not in self:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {attr!r}."
            )
        return getattr(self, f"_{attr}")

    def __setattr__(self, attr, value):
        if attr in self:
            attr = f"_{attr}"
            for class_ in self._classes:
                setattr(class_, attr, value)
        super().__setattr__(attr, value)

    def __iter__(self):
        yield from self._defaults

    def __dir__(self):
        return dir(self.__class__) + list(self)
