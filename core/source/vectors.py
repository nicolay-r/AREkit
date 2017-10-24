import io

class NewsVectorizedRelations:

    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    @staticmethod
    def from_file(filepath, labeled=False):
        """ Read the vectors from *.vectors.txt file
        """
        X = []
        labels = []
        with io.open(filepath, 'r') as f:
            for line in f.readlines():
                args = line.split()

                x = [float(a) for a in args]

                if (labeled):
                    y = x[len(x)-1]
                    x = x[:len(x)-1]
                    labels.append(y)

                assert(type(x) == list)

                X.append(x)

        return NewsVectorizedRelations(X, labels)
