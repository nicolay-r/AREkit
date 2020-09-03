class NamedEntityRecognition:

    separator = '-'
    begin_tag = 'B'
    inner_tag = 'I'

    def extract(self, sequences, return_single=True):
        return self.__extract(sequences=sequences,
                              return_single=return_single)

    def __extract(self, sequences, return_single):
        assert(isinstance(sequences, list))
        assert(isinstance(return_single, bool))

        seqs_tags = self._extract_tags(sequences)

        assert(len(sequences) == len(seqs_tags))

        lengths = [None] * len(sequences)
        positions = [None] * len(sequences)
        types = [None] * len(sequences)

        for s_ind, seq in enumerate(sequences):

            seq_tags = seqs_tags[s_ind]

            lengths[s_ind] = [len(entry) for entry in self.__merge(seq, seq_tags)]
            types[s_ind] = [self.__tag_type(tag) for tag in seq_tags if self.__tag_part(tag) == self.begin_tag]
            positions[s_ind] = [j for j, tag in enumerate(seq_tags) if self.__tag_part(tag) == self.begin_tag]

        if len(positions) == 1 and return_single:
            return positions[0], lengths[0], types[0]
        else:
            return positions, lengths, types

    @property
    def InputLimitation(self):
        raise NotImplementedError()

    # region tags info

    @property
    def LocationTypeStr(self):
        raise NotImplementedError()

    def GeoPoliticalTypeStr(self):
        raise NotImplementedError()

    @property
    def PersonTypeStr(self):
        raise NotImplementedError()

    @property
    def OrganizationTypeStr(self):
        raise NotImplementedError

    # endregion

    @property
    def NeedLemmatization(self):
        raise NotImplementedError()

    @property
    def NeedLowercase(self):
        raise NotImplementedError()

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
        return tag if NamedEntityRecognition.separator not in tag \
            else tag[:tag.index(NamedEntityRecognition.separator)]

    @staticmethod
    def __tag_type(tag):
        assert(isinstance(tag, str))
        return "" if NamedEntityRecognition.separator not in tag \
            else tag[tag.index(NamedEntityRecognition.separator) + 1:]

    # endregion
