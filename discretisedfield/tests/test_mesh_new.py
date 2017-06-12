import operator
import hypothesis
import hypothesis.strategies as st
import discretisedfield as df


@st.composite
def floats(draw):
    return draw(st.floats(min_value=1e-12, max_value=1e3,
                          allow_nan=False, allow_infinity=False))


@st.composite
def tuples(draw, elements=floats):
    return tuple(draw(st.lists(elements=elements(), max_size=3, min_size=3)))


@st.composite
def integers(draw):
    return draw(st.integers(min_value=1, max_value=1e6))


@st.composite
def mesh_args(draw):
    p1 = draw(tuples())
    p2 = draw(tuples().filter(lambda p2: all(map(operator.sub, p1, p2))))
    n = draw(st.tuples(integers(), integers(), integers()))
    cell = tuple(abs(a-b)/c for a, b, c in zip(p1, p2, n))
    return p1, p2, n, cell


@hypothesis.given(mesh_args())
def test_mesh(args):
    p1, p2, _, cell = args

    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    assert isinstance(mesh.pmin, tuple) and len(mesh.pmin) == 3
    assert isinstance(mesh.pmax, tuple) and len(mesh.pmax) == 3
