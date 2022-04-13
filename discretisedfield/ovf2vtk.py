"""OVF to VTK file conversion."""
import argparse

import discretisedfield as df


def ovf2vtk():
    """OVF to VTK conversion function.

    This method is used for command-line conversion of OVF files to VTK.

    """
    parser = argparse.ArgumentParser(
        prog="ovf2vtk", description="ovf2vtk - OVF to VTK file format conversion."
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True, help="Input OVF file(s)."
    )
    parser.add_argument(
        "--output", "-o", nargs="+", required=False, help="Output VTK file(s)."
    )
    args = parser.parse_args()

    if args.output:
        if len(args.input) != len(args.output):
            msg = (
                f"The number of input files ({len(args.input)}) does not "
                f"match the number of output files ({len(args.output)})."
            )
            raise ValueError(msg)
    else:
        # Output filenames are not provided and they are generated
        # automatically.
        args.output = [f"{filename[:-4]}.vtk" for filename in args.input]

    for input_file, output_file in zip(args.input, args.output):
        field = df.Field.fromfile(input_file)
        field.write(output_file)


if __name__ == "__main__":
    ovf2vtk()
