import os
import os.path
import sys


"""
Simple code to transform a teacher version of a notebook into a student
version, removing the solutions to the proposed exercises.

Usage:

    python generate-student.py [path]

where path is the path to the input file or files:

    - If path points to a teacher file, it will be transformed
    - If path is a directory, all teacher notebooks in

If no path is provided, it is requested.

The name of the teacher notebook must end with the suffix contained in variable
sufixsrd (which is _professor.ipynb at the time of writing these lines). All
notebooks with a different extension will be ignored.
"""


def transform_nb(dirpath, src_fname, tg_fname):
    """
    Transforms teacher notebook version in src_fname in folder dirpath into a
    student version, wich is saved in tg_fname in the same folder.

    To do so:
        - All code between <SOL> and </SOL> is removed
        - The line right below any line containing <FILL IN> is removed

    After that, html versions of scripts are created.
    """

    srcfile = os.path.join(dirpath, src_fname)
    tgfile = os.path.join(dirpath, tg_fname)

    with open(srcfile, 'r') as fin:
        with open(tgfile, 'w') as fout:

            state = True
            skip_next = False

            for line in fin:

                if state:

                    if '<FILL IN>' in line:
                        skip_next = True
                        fout.write(line)
                    else:
                        if skip_next:
                            # This line is ignored, because the above line
                            # contains a <FILL IN>
                            skip_next = False
                            if not line.endswith(',\n'):
                                # This is to avoid problems when the line to
                                # remove is the last line in its cell
                                fout.write('" "\n')
                        else:
                            fout.write(line)

                    if '<SOL>' in line:
                        state = False
                else:
                    if '</SOL>' in line:
                        fout.write('\n'+line)
                        state = True

    f = srcfile.replace(' ', '\ ')
    os.system('jupyter nbconvert --to html ' + f + ' --output ' +
              src_fname.replace('.ipynb', '.html'))

    # Clean student version
    os.system('jupyter nbconvert --ClearOutputPreprocessor.enabled=True ' +
              '--inplace ' + tgfile)

    # f = tgfile.replace(' ', '\ ')
    # os.system('jupyter nbconvert --to html ' + f + ' --output ' +
    #           tg_fname.replace('.ipynb', '.html'))
    # f = tgfile.replace(' ', '\ ')

    os.system('jupyter nbconvert --to slides ' + f)
    os.system('jupyter nbconvert --to pdf ' + f + ' --output ' +
              src_fname.replace('.ipynb', '.pdf'))

    return

# Configurable variables
sufixsrc = '_professor.ipynb'
sufixtg = '_student.ipynb'

# Read and check datapath
if len(sys.argv) > 1:
    datapath = sys.argv[1]
    # Check if project folder exists.
    if not os.path.exists(datapath):
        sys.exit("Data path does not exist.")
else:
    datapath = raw_input("Select the (absolute or relative) path to " +
                         "the data source (file or folder): ")

# File processing
if os.path.isfile(datapath):

    # Transform a single file
    if not datapath.endswith(sufixsrc):
        sys.exit("Data file does not end with " + sufixsrc)

    dirpath, src_fname = os.path.split(datapath)
    tg_fname = src_fname.replace(sufixsrc, sufixtg)
    print('Processing file:', src_fname)
    print('Target file: ', tg_fname)
    transform_nb(dirpath, src_fname, tg_fname)

else:

    # Transform all teacher files in a directory tree.
    for dirpath, dirnames, filenames in os.walk(datapath):

        for src_fname in [f for f in filenames if f.endswith(sufixsrc)]:

            tg_fname = src_fname.replace(sufixsrc, sufixtg)
            print('Processing file: ', src_fname)
            print('Target file: ', tg_fname)
            transform_nb(dirpath, src_fname, tg_fname)

