class NamedEntityRecognition:

    separator = '-'
    begin_tag = 'B'
    inner_tag = 'I'

    def extract(self, terms):

        tags = self._extract_tags(terms=terms)

        assert(len(terms) == len(tags))

        merged_terms = self.__merge(terms, tags)
        types = [self.__tag_type(tag) for tag in tags if self.__tag_part(tag) == self.begin_tag]
        positions = [i for i, tag in enumerate(tags) if self.__tag_part(tag) == self.begin_tag]

        return merged_terms, types, positions

    @property
    def NeedLemmatization(self):
        raise NotImplementedError()

    @property
    def NeedLowercase(self):
        raise NotImplementedError()

    def _extract_tags(self, terms):
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
