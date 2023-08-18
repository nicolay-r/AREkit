from collections import OrderedDict

from arekit.common.experiment.data_type import DataType


def create_fixed_folding(train_filenames, dev_filenames, test_filenames, limit=None):
    """ Create fixed data-folding based on the predefined list of filenames,
        written in file.
    """
    assert(isinstance(train_filenames, list))
    assert(isinstance(dev_filenames, list))
    assert(isinstance(test_filenames, list))

    filenames_by_ids = create_filenames_by_ids(filenames=train_filenames + dev_filenames + test_filenames)

    ids_by_filenames = {}
    for doc_id, filename in filenames_by_ids.items():
        ids_by_filenames[filename] = doc_id

    train_filenames = train_filenames if limit is None else train_filenames[:limit]
    test_filenames = test_filenames if limit is None else test_filenames[:limit]
    dev_filenames = dev_filenames if limit is None else dev_filenames[:limit]

    fixed_folding = {
        DataType.Train: [ids_by_filenames[filename] for filename in train_filenames],
        DataType.Test: [ids_by_filenames[filename] for filename in test_filenames],
        DataType.Dev: [ids_by_filenames[filename] for filename in dev_filenames]
    }

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

        assert(doc_id not in filenames_by_ids)
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
