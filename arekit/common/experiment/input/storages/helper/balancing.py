import logging
import pandas as pd
import random

from arekit.common.experiment import const


logger = logging.getLogger(__name__)


class SampleRowBalancerHelper(object):

    # region private methods

    @staticmethod
    def __get_class(df, uint_label):
        return df[df[const.LABEL] == uint_label]

    @staticmethod
    def __get_class_sizes(df, uint_labels):
        return [len(SampleRowBalancerHelper.__get_class(df=df, uint_label=uint_label))
                for uint_label in uint_labels]

    @staticmethod
    def __fill_blank(label_df, blank_df, seed=1):
        """
        Composes a DataFrame which has the same amount of examples as one with 'other_label'
        """
        random.seed(seed)
        labels_count = len(label_df)

        if labels_count > 0:
            for row_index in range(len(blank_df)):
                keep_row_index = row_index if row_index < labels_count else random.randint(0, labels_count - 1)
                row = label_df.iloc[keep_row_index]
                for column, value in row.items():
                    blank_df.at[row_index, column] = value

        return blank_df

    # endregion

    @staticmethod
    def calculate_balanced_df(df, create_blank_df, output_labels_uint):
        """
        Balancing related dataframe by amount of examples per class
        create_blank_df: func(size) -> df
        """
        assert(isinstance(df, pd.DataFrame))
        assert(isinstance(output_labels_uint, list))
        assert(callable(create_blank_df))

        original_class_sizes = SampleRowBalancerHelper.__get_class_sizes(df=df, uint_labels=output_labels_uint)

        larges_class_size = max(original_class_sizes)

        balanced = [SampleRowBalancerHelper.__fill_blank(
                        label_df=SampleRowBalancerHelper.__get_class(df=df, uint_label=uint_label),
                        blank_df=create_blank_df(larges_class_size))
                    for label_ind, uint_label in enumerate(output_labels_uint)
                    if original_class_sizes[label_ind] > 0]

        balanced_df = pd.concat(balanced)

        balanced_df.sort_values(by=[const.ID],
                                inplace=True,
                                ascending=True)

        balanced_class_sizes = SampleRowBalancerHelper.__get_class_sizes(df=balanced_df,
                                                                         uint_labels=output_labels_uint)

        logger.info("Rows count for uint labels [original]: {}".format(
            ["{} ({})".format(str(class_uint), str(count))
             for class_uint, count in enumerate(original_class_sizes) if count > 0]))

        logger.info("Rows count for uint labels [balanced]: {}".format(
            ["{} ({})".format(str(class_uint), str(count))
             for class_uint, count in enumerate(balanced_class_sizes) if count > 0]))

        logger.info("Rows count [balanced, total]: {}".format(sum(balanced_class_sizes)))

        return balanced_df

    # endregion