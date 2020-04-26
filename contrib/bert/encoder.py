from arekit.contrib.bert.format.opinions_io import OpinionsFormatter
from arekit.contrib.bert.format.samples_io import create_and_save_samples_to_tsv
from arekit.common.experiment.base import BaseExperiment
from arekit.common.experiment.opinions import extract_text_opinions
from arekit.common.experiment.data_type import DataType


class BertEncoder(object):

    @staticmethod
    def to_tsv(experiment):
        assert(isinstance(experiment, BaseExperiment))

        terms_per_context = 50

        for data_type in DataType.iter_supported():
            experiment.DataIO.NeutralAnnotator.create_collection(data_type)

        for data_type in DataType.iter_supported():

            text_opinions = extract_text_opinions(
                experiment=experiment,
                data_type=data_type,
                terms_per_context=terms_per_context)

            #
            # Compose csv file with related opinions (Necessary for evaluation)
            #
            OpinionsFormatter.create_and_save_opinions_to_csv(text_opinions=text_opinions,
                                                              data_type=data_type,
                                                              experiment=experiment)

            #
            # Train/Test input samples for bert
            #
            create_and_save_samples_to_tsv(text_opinions=text_opinions,
                                           data_type=data_type,
                                           experiment=experiment)
