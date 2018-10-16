class NamedEntityRecognition:

    def extract(self, text, merge=False):
        """
        text: unicode
            text
        merge: bool
            merge multiword entities into single list of terms

        returns: list
            list of terms (in case merge = False)
            or list of list of terms (when merge = True)
        """
        pass
