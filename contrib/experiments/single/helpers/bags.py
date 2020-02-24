from arekit.networks.context.training.bags.collection import BagsCollection


class BagsCollectionHelper:

    def __init__(self, bags_collection, name):
        assert(isinstance(bags_collection, BagsCollection))
        assert(isinstance(name, unicode))
        self.__bags_collection = bags_collection
        self.__name = name

    def print_log_statistics(self):
        print "Bags collection: {}".format(self.__name)
        print "\tBags count: {}".format(len(self.__bags_collection))
        print "\tSamples count: {}".format(sum(len(bag) for bag in self.__bags_collection))
        self.__print_label_statistics()

    def __print_label_statistics(self):
        labels_dict = {}
        for bag in self.__bags_collection:
            ls = bag.BagLabel.to_str()
            if ls not in labels_dict:
                labels_dict[ls] = 0
            labels_dict[ls] += 1

        items = list(labels_dict.iteritems())
        total = sum([class_count for class_count in labels_dict.itervalues()])
        for label_string, count in reversed(sorted(items, key=lambda itm: itm[1])):
            print u"\t{} -- {} ({}%)".format(label_string, count, round(100.0 * count / total, 2))
