import pytest
import numpy as np
import discretisedfield as df


def check_region(region):
    assert isinstance(region.p1, tuple)
    assert len(region.p1) == 3
    assert isinstance(region.p2, tuple)
    assert len(region.p2) == 3
    assert isinstance(region.pmin, tuple)
    assert len(region.pmin) == 3
    assert region.pmin in region
    assert isinstance(region.pmax, tuple)
    assert len(region.pmax) == 3
    assert region.pmax in region
    assert isinstance(region.l, tuple)
    assert len(region.l) == 3
    assert isinstance(region.volume, float)
    assert isinstance(region.centre, tuple)
    assert len(region.centre) == 3
    assert region.centre in region
    assert isinstance(region.random_point(), tuple)
    assert len(region.random_point()) == 3
    assert region.random_point() in region
    assert isinstance(repr(region), str)
    assert 'Region' in repr(region)
    assert region == region
    assert not region != region

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

    def test_init_valid_args(self):
        for p1, p2 in self.valid_args:
            region = df.Region(p1=p1, p2=p2)
            check_region(region)

    def test_init_invalid_args(self):
        for p1, p2 in self.invalid_args:
            with pytest.raises((TypeError, ValueError)):
                region = df.Region(p1=p1, p2=p2)  # Raised by descriptors.

    def test_zero_edge_length(self):
        args = [[(0, 100e-9, 1e-9), (150e-9, 100e-9, 6e-9)],
                [(0, 101e-9, -1), (150e-9, 101e-9, 0)],
                [(10e9, 10e3, 0), (0.01e12, 11e3, 5)]]

        for p1, p2 in args:
            with pytest.raises(ValueError) as excinfo:
                region = df.Region(p1=p1, p2=p2)
            assert 'is zero' in str(excinfo.value)

    def test_pmin_pmax_l_centre_volume(self):
        p1 = (0, -4, 16.5)
        p2 = (15, -6, 11)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        assert region.pmin == (0, -6, 11)
        assert region.pmax == (15, -4, 16.5)
        assert region.l == (15, 2, 5.5)
        assert region.centre == (7.5, -5, 13.75)
        assert region.volume == 165

        p1 = (-10, 0, 0)
        p2 = (10, 1, 1)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        assert region.pmin == (-10, 0, 0)
        assert region.pmax == (10, 1, 1)
        assert region.l == (20, 1, 1)
        assert region.centre == (0, 0.5, 0.5)
        assert region.volume == 20

        p1 = (-18.5e-9, 10e-9, 0)
        p2 = (10e-9, 5e-9, -10e-9)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        assert np.allclose(region.pmin, (-18.5e-9, 5e-9, -10e-9))
        assert np.allclose(region.pmax, (10e-9, 10e-9, 0))
        assert np.allclose(region.l, (28.5e-9, 5e-9, 10e-9))
        assert np.allclose(region.centre, (-4.25e-9, 7.5e-9, -5e-9))
        assert abs(region.volume - 1425 * (1e-9**3)) < 1e-30

    def test_repr(self):
        p1 = (-1, -4, 11)
        p2 = (15, 10.1, 12.5)
        region = df.Region(p1=p1, p2=p2)

        rstr = 'Region(p1=(-1.0, -4.0, 11.0), p2=(15.0, 10.1, 12.5))'
        assert repr(region) == rstr

    def test_eq_ne(self):
        region1 = df.Region(p1=(0, 0, 0), p2=(10, 10, 10))
        region2 = df.Region(p1=(0, 0, 0), p2=(10, 10, 10))
        region3 = df.Region(p1=(3, 3, 3), p2=(10, 10, 10))

        assert region1 == region2
        assert not region1 != region2
        assert region1 != region3
        assert not region1 == region3

    def test_contains(self):
        p1 = (0, 10e-9, 0)
        p2 = (10e-9, 0, 20e-9)
        region = df.Region(p1=p1, p2=p2)

        assert (0, 0, 0) in region
        assert (10e-9, 10e-9, 20e-9) in region
        assert (5e-9, 5e-9, 5e-9) in region
        assert (11e-9, 11e-9, 11e-9) not in region
        assert (-1e-9, -1e-9, -1e-9) not in region
