import discretisedfield as df


def test_interact():
    p1 = (-50e9, -50e9, -50e9)
    p2 = (50e9, 50e9, 50e9)
    n = (10, 10, 10)
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
    field = df.Field(mesh, dim=3, value=(1, 2, 0))

    # Only test whether it runs.
    @df.interact(x=field.mesh.slider('x'))
    def myplot(x):
        field.plane(x=x).mpl()
