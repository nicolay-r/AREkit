from os.path import join


def join_dir_with_subfolder_name(subfolder_name, dir):
    """ Returns subfolder in in directory
    """
    assert(isinstance(subfolder_name, str))
    assert(isinstance(dir, str))

    target_dir = join(dir, "{}/".format(subfolder_name))
    return target_dir