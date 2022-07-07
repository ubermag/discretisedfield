import json

import numpy as np

import discretisedfield as df


class RegionJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, df.Region):
            return o.to_dict()
        elif isinstance(o, np.int64):
            return int(o)
        elif isinstance(o, np.float64):
            return float(o)
        else:
            super().default(o)


def strip_extension(filename):
    """Strip the file extension from a file name (which can be a full path)."""
    return "".join(filename.split(".")[:-1])
