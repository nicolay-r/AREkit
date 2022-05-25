from arekit.common.utils import progress_bar_defined


def iter_synonym_groups(input_file, sep=",", desc=""):
    """ All the synonyms groups organized in lines, separated by `sep`
    """
    lines = input_file.readlines()

    lines_it = progress_bar_defined(lines,
                                    total=len(lines),
                                    desc=desc,
                                    unit="opins")

    for line in lines_it:

        if isinstance(line, bytes):
            line = line.decode()

        yield line.split(sep)
