class NamedEntityRecognition:

    def extract(self, text, merge=False):
        """
        text: unicode
            text
        merge: bool
            merge multiword entities into single list of parsed_news

        returns: list
            list of parsed_news (in case merge = False)
            or list of list of parsed_news (when merge = True)
        """
        pass
