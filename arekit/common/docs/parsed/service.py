from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.docs.parsed.providers.base import BaseParsedDocumentServiceProvider


class ParsedDocumentService(object):
    """ Represents a collection of providers, combined with the parsed doc.
    """

    def __init__(self, parsed_doc, providers):
        assert(isinstance(parsed_doc, ParsedDocument))
        assert(isinstance(providers, list))
        self.__parsed_doc = parsed_doc
        self.__providers = {}

        for provider in providers:
            assert(isinstance(provider, BaseParsedDocumentServiceProvider))
            assert(provider.Name not in self.__providers)

            # Link provider with the related name.
            self.__providers[provider.Name] = provider

            # Post initialize with the related parsed doc.
            provider.init_parsed_doc(self.__parsed_doc)


    @property
    def ParsedDocument(self):
        return self.__parsed_doc

    def get_provider(self, name):
        return self.__providers[name]
