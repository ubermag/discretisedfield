import argparse
import discretisedfield as df


def convert_files(input_files, output_files):
    for input_file, output_file in zip(input_files, output_files):
        field = df.Field.fromfile(input_file)
        field.write(output_file)


def main():
    parser = argparse.ArgumentParser(
        prog='ovf2vtk',
        description='ovf2vtk - ovf to VTK format conversion'
    )
    parser.add_argument('--infile', type=argparse.FileType('r'),
                        help='One or more input files', nargs='+',
                        required=True)
    parser.add_argument('--outfile', type=argparse.FileType('w'), nargs='+',
                        help='One or more output files, optional')
    args = parser.parse_args()

    if args.outfile:
        if len(args.infile) == len(args.outfile):
            input_files = [f.name for f in args.infile]
            output_files = [f.name for f in args.outfile]
        else:
            msg = 'The number of input and output files do not match.'
            raise ValueError(msg)
    else:
        input_files = [f.name for f in args.infile]
        output_files = [f'{f.split(".")[0]}.vtk' for f in input_files]

    convert_files(input_files, output_files)


if __name__ == "__main__":
    main()
