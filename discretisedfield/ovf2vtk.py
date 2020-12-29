"""OVF to VTK file conversion"""
import argparse
import discretisedfield as df


def ovf2vtk():
    """OVF to VTK conversion function.

    This method is used for command-line conversion of OVF files to VTK.

    """
    parser = argparse.ArgumentParser(
        prog='ovf2vtk',
        description='ovf2vtk - OVF to VTK file format conversion.'
    )
    parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                        nargs='+', required=True, help='Input OVF file(s).')
    parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                        nargs='+', required=False, help='Output VTK file(s).')
    args = parser.parse_args()

    input_files = [f.name for f in args.input]

    if args.output:
        # Output filenames provided.
        if len(args.input) == len(args.output):
            output_files = [f.name for f in args.output]
        else:
            msg = (f'The number of input files ({len(args.input)}) does not '
                   f'match the number of output files ({len(args.output)}).')
            raise ValueError(msg)
    else:
        # Output filenames are not provided and they are generated
        # automatically.
        output_files = [f'{filename[:-4]}.vtk' for filename in input_files]

    for input_file, output_file in zip(input_files, output_files):
        field = df.Field.fromfile(input_file)
        field.write(output_file)


if __name__ == '__main__':
    ovf2vtk()
