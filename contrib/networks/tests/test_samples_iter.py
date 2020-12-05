import pandas as pd
import gzip
import sys
import unittest

sys.path.append('../../../')

from arekit.common.experiment import const
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample


class TestSamplesIteration(unittest.TestCase):

    def test_simple(self):
        vocab_filepath = u"test_data/vocab.txt.gz"
        samples_filepath = u"test_data/sample_train.tsv.gz"
        words_vocab = self.__read_vocab(vocab_filepath)
        config = DefaultNetworkConfig()
        config.modify_terms_per_context(50)

        self.__test_core(words_vocab=words_vocab,
                         config=config,
                         samples_filepath=samples_filepath,
                         has_title=True)

    # region private methods

    @staticmethod
    def __iter_tsv_gzip(input_file):
        """Reads a tab separated value file."""
        df = pd.read_csv(input_file,
                         compression='gzip',
                         sep='\t',
                         encoding='utf-8')

        for row_index, _ in enumerate(df[const.ID]):
            yield df.iloc[row_index]

    @staticmethod
    def __read_vocab(input_file):
        words = {}
        with gzip.open(input_file, mode="rt") as f:
            for w_ind, line in enumerate(f.readlines()):
                w = line.decode('utf-8').strip()
                if w in words:
                    raise Exception(u"Word already presented: {}".format(w).encode('utf-8'))
                words[w] = w_ind
        return words

    def __terms_to_text_line(self, terms, frame_inds_set):
        assert(isinstance(frame_inds_set, set))
        words = []
        for t_index, t_value in enumerate(terms):
            value = t_value
            if t_index in frame_inds_set:
                value = u"[[{}]]".format(t_value)
            words.append(value)
        return u" ".join(words)

    def __test_core(self, words_vocab, config, samples_filepath, has_title):
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(samples_filepath, unicode))

        samples = []
        labels_scaler = ThreeLabelScaler()
        for i, row in enumerate(self.__iter_tsv_gzip(input_file=samples_filepath)):

            row = ParsedSampleRow(row, labels_scaler=labels_scaler)

            print u"------------------"
            print u"INPUT SAMPLE DATA"
            print u"------------------"
            print u"id: {}".format(row.SampleID)
            print u"label: {}".format(row.Sentiment)
            print u"subj_ind: {}".format(row.SubjectIndex)
            print u"obj_ind: {}".format(row.ObjectIndex)
            print u"frame_inds: {}".format(row.TextFrameVariantIndices)
            print u"frame_roles_uint: {}".format(row.TextFrameVariantRoles)
            print u"syn_obj: {}".format(row.SynonymObjectInds)
            print u"syn_subj: {}".format(row.SynonymSubjectInds)
            print u"terms:".format(row.Terms)

            print self.__terms_to_text_line(terms=row.Terms, frame_inds_set=set(row.TextFrameVariantIndices))

            sample = InputSample.create_from_parameters(
                input_sample_id=row.SampleID,
                terms=row.Terms,
                subj_ind=int(row.SubjectIndex),
                obj_ind=int(row.ObjectIndex),
                words_vocab=words_vocab,
                is_external_vocab=True,
                pos_tagger=config.PosTagger,
                terms_per_context=config.TermsPerContext,
                frames_per_context=config.FramesPerContext,
                synonyms_per_context=config.SynonymsPerContext,
                frame_inds=row.TextFrameVariantIndices,
                frame_sent_roles=row.TextFrameVariantRoles,
                syn_subj_inds=row.SynonymSubjectInds,
                syn_obj_inds=row.SynonymObjectInds)

            print u"------------------"
            print u"NETWORK INPUT DATA"
            print u"------------------"
            for key, value in sample:
                print u"{key}:\n{value}".format(key=key, value=value)

            samples.append(sample)

    # endregion


if __name__ == '__main__':
    unittest.main()
