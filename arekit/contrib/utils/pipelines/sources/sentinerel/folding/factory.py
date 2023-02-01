from arekit.contrib.utils.pipelines.sources.sentinerel.folding.fixed import create_fixed_folding


class SentiNERELFoldingFactory:
    """ Factory of the variety types of the splits that
        are considered within the present experiments.
    """

    @staticmethod
    def create_fixed_folding(fixed_split_filepath, limit=None):
        """
            fixed_split_filepath: str
                filepath to the fixed collection split.
            limit: int
                Allows to limit amount of documents (utilized for testing reasons)
        """

        train_filenames, test_filenames = SentiNERELFoldingFactory._read_train_test(fixed_split_filepath)
        if limit is not None:
            train_filenames = train_filenames[:limit]
            test_filenames = test_filenames[:limit]
        filenames_by_ids, data_folding = create_fixed_folding(train_filenames=train_filenames,
                                                              test_filenames=test_filenames)

        return filenames_by_ids, data_folding

    @staticmethod
    def _read_train_test(filepath):
        with open(filepath, "r") as f:
            parts = []
            for line in f.readlines():
                parts.append(line.strip().split(','))
        return parts[0], parts[1]
