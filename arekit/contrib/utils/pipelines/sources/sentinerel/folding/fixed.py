from collections import OrderedDict

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.fixed import FixedFolding


def create_fixed_folding(train_filenames, test_filenames):
    """ Create fixed data-folding based on the predefined list of filenames,
        written in file.
    """
    assert(isinstance(train_filenames, list))
    assert(isinstance(test_filenames, list))

    filenames_by_ids = create_filenames_by_ids(filenames=train_filenames + test_filenames)

    ids_by_filenames = {}
    for doc_id, filename in filenames_by_ids.items():
        ids_by_filenames[filename] = doc_id

    train_doc_ids = [ids_by_filenames[filename] for filename in train_filenames]
    test_doc_ids = [ids_by_filenames[filename] for filename in test_filenames]

    fixed_folding = FixedFolding.from_parts({
        DataType.Train: train_doc_ids,
        DataType.Test: test_doc_ids,
        DataType.Etalon: test_doc_ids,
        DataType.Dev: test_doc_ids
    })

    return filenames_by_ids, fixed_folding


def create_filenames_by_ids(filenames):
    """ Indexing filenames
    """

    def __create_new_id(default_id):
        new_id = default_id
        while new_id in filenames_by_ids:
            new_id += 1
        return new_id

    default_id = 0

    filenames_by_ids = OrderedDict()
    for fname in filenames:

        doc_id = number_from_string(fname)

        if doc_id is None:
            doc_id = __create_new_id(default_id)
            default_id = doc_id

        filenames_by_ids[doc_id] = fname

    return filenames_by_ids


def number_from_string(s):
    assert(isinstance(s, str))

    digit_chars_prefix = []

    for chr in s:
        if chr.isdigit():
            digit_chars_prefix.append(chr)
        else:
            break

    if len(digit_chars_prefix) == 0:
        return None

    return int("".join(digit_chars_prefix))
