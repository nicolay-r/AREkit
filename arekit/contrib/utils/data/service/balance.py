import gc
import importlib

from arekit.contrib.utils.data.storages.pandas_based import PandasBasedRowsStorage


class PandasBasedStorageBalancing(object):

    @staticmethod
    def create_balanced_from(storage, column_name, free_origin=True):
        """ Performs oversampled balancing.

            Note: it is quite important to remove previously created storage
            in order to avoid memory leaking.

            storage: PandasBasedRowsStorage
                storage contents to be balanced.

            column_name: str
                column utilized for balancing.

            free_origin: bool
                indicates whether there is a need to release the resources
                utilized for the original storage.
        """
        assert(isinstance(storage, PandasBasedRowsStorage))

        original_df = storage.DataFrame

        max_size = original_df[column_name].value_counts().max()

        dframes = []
        for class_index, group in original_df.groupby(column_name):
            dframes.append(group.sample(max_size - len(group), replace=True))

        # Clear resources.
        pd = importlib.import_module("pandas")
        balanced_df = pd.concat(dframes + [original_df])

        # Removing temporary created dataframe.
        for df in dframes:
            del df

        # Marking the original dataframe as released
        # in terms of the allocated memory for it.
        if free_origin:
            storage.free()

        gc.collect()

        return PandasBasedRowsStorage(df=balanced_df)
