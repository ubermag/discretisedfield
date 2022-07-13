import json
import pathlib

import numpy as np

import discretisedfield as df


class _RegionJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, df.Region):
            return o.to_dict()
        elif isinstance(o, np.int64):
            return int(o)
        elif isinstance(o, np.float64):
            return float(o)
        else:
            super().default(o)


def strip_extension(filename: pathlib.Path):
    """Strip the file extension from a file name (which can be a full path)."""
    return str(filename)[: -len(filename.suffix)]
