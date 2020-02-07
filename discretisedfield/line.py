import numbers
import numpy as np
import seaborn as sns
import ubermagutil.units as uu
import matplotlib.pyplot as plt


class Line:
    def __init__(self, points, values):
        self.points = points
        self.values = values

    @property
    def length(self):
        r_vector = np.subtract(self.points[-1], self.points[0])
        return np.linalg.norm(r_vector)

    @property
    def n(self):
        return len(self.points)

    @property
    def dim(self):
        if isinstance(self.values[0], numbers.Real):
            return 1
        else:
            return len(self.values[0])

    def __repr__(self):
        return 'Line(...)'

    def mpl(self, ax=None, figsize=None, multiplier=None, **kwargs):
        sns.set()
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        if multiplier is None:
            multiplier = uu.si_multiplier(self.length)

        x_array = np.linspace(0, self.length, self.n)
        x_array = np.divide(x_array, multiplier)

        if self.dim == 1:
            with sns.axes_style('darkgrid'):
                ax.plot(x_array, self.values, **kwargs)
        else:
            vals = list(zip(*self.values))
            for val, label in zip(vals, 'xyz'):
                with sns.axes_style('darkgrid'):
                    ax.plot(x_array, val, label=label, **kwargs)
            ax.legend()

        ax.set_xlabel(f'r ({uu.rsi_prefixes[multiplier]}m)')
        ax.set_ylabel('value')
