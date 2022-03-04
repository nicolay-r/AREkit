from arekit.processing.entities.obj_desc import NerObjectDescriptor


class BaseNER(object):
    """ CoNLL format based Named Entity Extractor
        for list of input sequences, where sequence is a list of terms.
    """

    separator = '-'
    begin_tag = 'B'
    inner_tag = 'I'

    def extract(self, sequences):
        return self.__extract_objects_core(sequences=sequences)

    def __extract_objects_core(self, sequences):
        assert(isinstance(sequences, list))
        seqs_tags = self._extract_tags(sequences)
        assert(len(sequences) == len(seqs_tags))

        extracted = []
        for sequence_ind, sequence in enumerate(sequences):
            seq_tags = seqs_tags[sequence_ind]
            objs_len = [len(entry) for entry in self.__merge(sequence, seq_tags)]
            objs_type = [self.__tag_type(tag) for tag in seq_tags if self.__tag_part(tag) == self.begin_tag]
            objs_positions = [j for j, tag in enumerate(seq_tags) if self.__tag_part(tag) == self.begin_tag]

            assert(len(objs_len) == len(objs_type) == len(objs_positions))

            seq_obj_descriptors = [NerObjectDescriptor(pos=objs_positions[i],
                                                       length=objs_len[i],
                                                       obj_type=objs_type[i])
                                   for i in range(len(objs_len))]

            extracted.append(seq_obj_descriptors)

        return extracted

    def _extract_tags(self, seqences):
        raise NotImplementedError()

    # region private methods

    def __merge(self, terms, tags):
        merged = []
        for i, tag in enumerate(tags):
            current_tag = self.__tag_part(tag)
            if current_tag == self.begin_tag:
                merged.append([terms[i]])
            elif current_tag == self.inner_tag and len(merged) > 0:
                merged[-1].append(terms[i])
        return merged

    @staticmethod
    def __tag_part(tag):
        assert(isinstance(tag, str))
        return tag if BaseNER.separator not in tag \
            else tag[:tag.index(BaseNER.separator)]

    @staticmethod
    def __tag_type(tag):
        assert(isinstance(tag, str))
        return "" if BaseNER.separator not in tag \
            else tag[tag.index(BaseNER.separator) + 1:]

    # endregion
