from arekit.contrib.bert.format.opinions_io import create_and_save_opinions_to_csv
from arekit.contrib.bert.format.samples_io import create_and_save_samples_to_tsv
from arekit.contrib.experiments.data_io import DataIO
from arekit.contrib.experiments.nn_io.rusentrel_with_ruattitudes import RuSentRelWithRuAttitudesBasedExperimentIO
from arekit.networks.data_type import DataType
from read_text_opinions import extract_text_opinions


def to_tsv(data_io):
    assert(isinstance(data_io, DataIO))

    model_name = u"bert"

    io = RuSentRelWithRuAttitudesBasedExperimentIO(
        data_io=data_io,
        model_name=model_name,
        cv_count=3)

    terms_per_context = 50

    for data_type in DataType.iter_supported():
        data_io.NeutralAnnontator.create_collection(data_type)

    for data_type in DataType.iter_supported():

        text_opinions = extract_text_opinions(
            io=io,
            data_type=data_type,
            terms_per_context=terms_per_context)

        #
        # Compose csv file with related opinions (Necessary for evaluation)
        #
        create_and_save_opinions_to_csv(text_opinions=text_opinions,
                                        data_type=data_type,
                                        model_name=model_name)

        #
        # Train/Test input samples for bert
        #
        create_and_save_samples_to_tsv(text_opinions=text_opinions,
                                       pnc=text_opinions.RelatedParsedNewsCollection,
                                       data_type=data_type,
                                       model_name=model_name)
