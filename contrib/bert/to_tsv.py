from arekit.contrib.bert.format.opinions_io import create_and_save_opinions_to_csv
from arekit.contrib.bert.format.samples_io import create_and_save_samples_to_tsv
from arekit.networks.data_type import DataType
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from readers.rusentrel_ds_io import RuSentRelWithRuAttitudesDataIO
from neutrals import RuSentRelNeutralAnnotationCreator
from read_text_opinions import extract_text_opinions


# TODO. Refactor as method.
if __name__ == "__main__":

    terms_per_context = 50
    stemmer = MystemWrapper()

    synonyms = RuSentRelSynonymsCollection.load_collection(
        stemmer=stemmer,
        is_read_only=True)

    # io = RuSentRelDataIO(synonyms)
    io = RuSentRelWithRuAttitudesDataIO(stemmer=stemmer, synonyms=synonyms)

    nac = RuSentRelNeutralAnnotationCreator(stemmer=stemmer,
                                            synonyms=synonyms)
    nac.create(True)
    nac.create(False)

    for data_type in [DataType.Train, DataType.Test]:

        text_opinions = extract_text_opinions(
            io=io,
            data_type=data_type,
            terms_per_context=terms_per_context)

        #
        # Compose csv file with related opinions (Necessary for evaluation)
        #
        create_and_save_opinions_to_csv(text_opinions=text_opinions,
                                        data_type=data_type,
                                        model_name=io.ModelName)

        #
        # Train/Test input samples for bert
        #
        create_and_save_samples_to_tsv(text_opinions=text_opinions,
                                       pnc=text_opinions.RelatedParsedNewsCollection,
                                       data_type=data_type,
                                       model_name=io.ModelName)
