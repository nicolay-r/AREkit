from os.path import basename


def __iter_filtered_filenames(filenames_iter):
    for filename in filenames_iter:
        extension = filename[-4:]
        # Crop extension.
        filename = filename[:-4]
        if extension != ".txt":
            continue
        yield filename, basename(filename)


def iter_filename_and_splittype(filenames_it, splits):
    for doc_id, data in enumerate(__iter_filtered_filenames(filenames_it)):
        filepath, filename = data
        for split_type, split_name in splits:
            if split_name in filepath:
                yield filename, split_type


def iter_collection_filenames(filenames_it):
    for doc_id, filename in enumerate(__iter_filtered_filenames(filenames_it)):
        yield doc_id, filename
