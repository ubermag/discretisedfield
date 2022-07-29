import json

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
