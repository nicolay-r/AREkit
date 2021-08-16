from os.path import dirname, join

import pandas as pd
import gzip
import sys
import unittest


sys.path.append('../../../')

from arekit.common.experiment import const
from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample


class TestSamplesIteration(unittest.TestCase):

    __show_examples = False
    __show_shifted_examples = False

    def __get_local_dir(self, local_filepath):
        return join(dirname(__file__), local_filepath)

    def test_check_all_samples(self):
        vocab_filepath = self.__get_local_dir("test_data/vocab.txt.gz")
        samples_filepath = self.__get_local_dir("test_data/sample-train.tsv.gz")
        words_vocab = self.__read_vocab(vocab_filepath)
        config = DefaultNetworkConfig()
        config.modify_terms_per_context(50)

        self.__test_core(words_vocab=words_vocab,
                         config=config,
                         samples_filepath=samples_filepath)

    def test_show_all_samples(self):
        self.__show_examples = True
        self.test_check_all_samples()

    def test_show_shifted_examples_only(self):
        self.__show_examples = True
        self.__show_shifted_examples = True
        self.test_check_all_samples()

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
                    raise Exception("Word already presented: {}".format(w).encode('utf-8'))
                words[w] = w_ind
        return words

    def __terms_to_text_line(self, terms, frame_inds_set):
        assert(isinstance(frame_inds_set, set))
        words = []
        for t_index, t_value in enumerate(terms):
            value = t_value
            if t_index in frame_inds_set:
                value = "[[{}]]".format(t_value)
            words.append(value)
        return " ".join(words)

    def __test_core(self, words_vocab, config, samples_filepath):
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(samples_filepath, str))

        samples = []
        for i, row in enumerate(self.__iter_tsv_gzip(input_file=samples_filepath)):

            # Perform row parsing process.
            row = ParsedSampleRow(row)

            subj_ind = row.SubjectIndex
            obj_ind = row.ObjectIndex

            sample = InputSample.create_from_parameters(
                input_sample_id=row.SampleID,
                terms=row.Terms,
                entity_inds=row.EntityInds,
                subj_ind=int(row.SubjectIndex),
                obj_ind=int(row.ObjectIndex),
                words_vocab=words_vocab,
                is_external_vocab=True,
                terms_per_context=config.TermsPerContext,
                frames_per_context=config.FramesPerContext,
                synonyms_per_context=config.SynonymsPerContext,
                frame_inds=row.TextFrameVariantIndices,
                frame_sent_roles=row.TextFrameVariantRoles,
                pos_tags=row.PartOfSpeechTags,
                syn_subj_inds=row.SynonymSubjectInds,
                syn_obj_inds=row.SynonymObjectInds)

            if sample._shift_index_dbg == 0 and self.__show_shifted_examples:
                continue

            if self.__show_examples:
                print("------------------")
                print("INPUT SAMPLE DATA")
                print("------------------")
                print("offset index (debug): {}".format(sample._shift_index_dbg))
                print("id: {}".format(row.SampleID))
                print("label: {}".format(row.UintLabel))
                print("entity_inds: {}".format(row.EntityInds))
                print("subj_ind: {}".format(subj_ind))
                print("obj_ind: {}".format(obj_ind))
                print("frame_inds: {}".format(row.TextFrameVariantIndices))
                print("frame_roles_uint: {}".format(row.TextFrameVariantRoles))
                print("syn_obj: {}".format(row.SynonymObjectInds))
                print("syn_subj: {}".format(row.SynonymSubjectInds))
                print("terms:".format(row.Terms))
                print("pos_tags:".format(row.PartOfSpeechTags))

                print(self.__terms_to_text_line(terms=row.Terms, frame_inds_set=set(row.TextFrameVariantIndices)))

                print("------------------")
                print("NETWORK INPUT DATA")
                print("------------------")
                for key, value in sample:
                    print("{key}:\n{value}".format(key=key, value=value))

            samples.append(sample)

    # endregion


if __name__ == '__main__':
    unittest.main()
