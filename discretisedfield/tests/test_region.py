import pytest
import numpy as np
import discretisedfield as df


def check_region(region):
    assert isinstance(region.pmin, tuple)
    assert len(region.pmin) == 3
    assert isinstance(region.pmax, tuple)
    assert len(region.pmax) == 3


class TestRegion:
    def setup(self):
        self.valid_args = [[(0, 0, 0), (5, 5, 5)],
                           [(-1, 0, -3), (5, 7, 5)],
                           [(0, 0, 0), (5e-9, 5e-9, 5e-9)],
                           [(-1.5e-9, -5e-9, 0), (1.5e-9, -15e-9, -10e-9)],
                           [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0))],
                           [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6)],
                           [(0, 125e-9, 0), (500e-9, 0, -3e-9)]]

        self.invalid_args = [[('1', 0, 0), (1, 1, 1)],
                             [(-1.5e-9, -5e-9, 'a'), (1.5e-9, 15e-9, 16e-9)],
                             [(-1.5e-9, -5e-9, 0), (1.5e-9, 16e-9)],
                             [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 1+2j)],
                             ['string', (5, 1, 1e-9)]]

    def test_init(self):
        p1 = (0, -4, 16.5)
        p2 = (15, -6, 11)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        assert region.pmin == (0, -6, 11)
        assert region.pmax == (15, -4, 16.5)

    def test_init_valid_args(self):
        for p1, p2 in self.valid_args:
            region = df.Region(p1=p1, p2=p2)

            check_region(region)

    def test_init_invalid_args(self):
        for p1, p2 in self.invalid_args:
            with pytest.raises((TypeError, ValueError)):
                # Exceptions are raised by descriptors.
                region = df.Region(p1=p1, p2=p2)

    def test_zero_domain_edge_length(self):
        args = [[(0, 100e-9, 1e-9), (150e-9, 100e-9, 6e-9)],
                [(0, 101e-9, -1), (150e-9, 101e-9, 0)],
                [(10e9, 10e3, 0), (0.01e12, 11e3, 5)]]

        for p1, p2 in args:
            with pytest.raises(ValueError) as excinfo:
                region = df.Region(p1=p1, p2=p2)
            assert 'is zero' in str(excinfo.value)

    def test_repr(self):
        p1 = (-1, -4, 11)
        p2 = (15, 10.1, 12.5)
        region = df.Region(p1=p1, p2=p2)

        rstr = 'Region(p1=(-1.0, -4.0, 11.0), p2=(15.0, 10.1, 12.5))'
        assert repr(region) == rstr

    def test_contains(self):
        p1 = (0, 10e-9, 0)
        p2 = (10e-9, 0, 20e-9)
        region = df.Region(p1=p1, p2=p2)

        assert (0, 0, 0) in region
        assert (10e-9, 10e-9, 10e-9) in region
        assert (5e-9, 5e-9, 5e-9) in region
        assert (11e-9, 11e-9, 11e-9) not in region
        assert (-1e-9, -1e-9, -1e-9) not in region
