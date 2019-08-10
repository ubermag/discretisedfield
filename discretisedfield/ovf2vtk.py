import argparse
import discretisedfield as df


def convert_files(inputs, outputs):
    for in_, out_ in zip(inputs, outputs):
        field = df.Field.fromfile(in_)
        field.write(out_)


def main():
    parser = argparse.ArgumentParser(
        prog='ovf2vtk',
        description='ovf2vtk - data conversion from omf/ovf file format to VTK'
    )
    parser.add_argument('--infile', type=argparse.FileType('r'),
                        help='One or more input file', nargs='+',
                        required=True)
    parser.add_argument('--outfile', type=argparse.FileType('w'), nargs='+',
                        help='One or more output file, optional')
    args = parser.parse_args()

    # check name of output file
    if args.outfile:

        # check count input and output files
        if len(args.infile) == len(args.outfile):
            input_files = [file.name for file in args.infile]
            output_files = [file.name for file in args.outfile]
        else:
            print('\nError: The number of input & output files does not match')
            return
    else:
        input_files = [file.name for file in args.infile]
        output_files = ['{}.vtk'.format(file.split('.')[0])
                        for file in input_files]

    convert_files(input_files, output_files)


if __name__ == "__main__":
    main()
