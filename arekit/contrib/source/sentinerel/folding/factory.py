from arekit.contrib.source.sentinerel.folding.fixed import create_fixed_folding_doc_ids


class SentiNERELFoldingFactory:
    """ Factory of the variety types of the splits that
        are considered within the present experiments.
    """

    @staticmethod
    def create_fixed_folding(file, limit=None):
        """ limit: int
                Allows to limit amount of documents (utilized for testing reasons)
        """

        train_filenames, test_filenames = SentiNERELFoldingFactory._read_train_test(f=file)
        if limit is not None:
            train_filenames = train_filenames[:limit]
            test_filenames = test_filenames[:limit]
        filenames_by_ids, data_folding = create_fixed_folding_doc_ids(train_filenames=train_filenames,
                                                                      test_filenames=test_filenames)

        return filenames_by_ids, data_folding

    @staticmethod
    def _read_train_test(f):
        parts = []
        for line in f.readlines():
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            parts.append(line.strip().split(','))
        return parts[0], parts[1]
