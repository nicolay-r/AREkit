from arekit.contrib.bert.format.opinions_io import create_and_save_opinions_to_csv
from arekit.contrib.bert.format.samples_io import create_and_save_samples_to_tsv
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.contrib.experiments.single.embedding.opinions import extract_text_opinions
from arekit.networks.data_type import DataType


def to_tsv(experiment_io):
    """
    experiment_io:
        Example of experiment io
        RuSentRelWithRuAttitudesBasedExperimentIO(data_io=data_io, model_name="bert")
    """
    assert(isinstance(experiment_io, BaseExperimentNeuralNetworkIO))

    terms_per_context = 50

    for data_type in DataType.iter_supported():
        experiment_io.DataIO.NeutralAnnotator.create_collection(data_type)

    for data_type in DataType.iter_supported():

        text_opinions = extract_text_opinions(
            experiment_io=experiment_io,
            data_type=data_type,
            terms_per_context=terms_per_context)

        #
        # Compose csv file with related opinions (Necessary for evaluation)
        #
        create_and_save_opinions_to_csv(text_opinions=text_opinions,
                                        data_type=data_type,
                                        experiment_io=experiment_io)

        #
        # Train/Test input samples for bert
        #
        create_and_save_samples_to_tsv(text_opinions=text_opinions,
                                       data_type=data_type,
                                       experiment_io=experiment_io)
