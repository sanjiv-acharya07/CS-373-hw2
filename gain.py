
from abc import ABC, abstractmethod
import numpy as np
import collections


class UncertaintyScorer(ABC):
    """This is the base class that describes the behavior of a scoring function
    that measures the amount of uncertainty (or 'purity') of the labels of a
    set of data points.

    This class implements some methods like(e.g.
    `compute_class_probs`), but leaves the main scoring functions to be
    implemented by the derived classes (e.g `InformationGain`).

    Attributes
    ----------
    class_labels : set
        This set contains all the unique possible class labels
    alpha : int
        Value used for Laplace smoothing
    """
    def __init__(self, class_labels, alpha=0):
        self.class_labels = set(class_labels)
        self.alpha = alpha

    @abstractmethod
    def score(self, y):
        """This function scores the uncertainty in the class labels of a set
        of examples. It will be implemented in derived classes.
        """
        ...

    @abstractmethod
    def compute_gain(self, X, y, j):
        """This function will compute the gain (e.g. Information
        Gain) for a dataset and a given attribute. It will be implemented in
        derived classes.
        """
        ...

    def compute_class_probs(self, y):
        """Compute a distribution over all possible labels, estimated with the
        given set of data points.

        If an empty array is given, assign equal (uniform) probability for
        all classes.

        To avoid issues with probabilities being 0, you must use Laplacian
        smoothing when computing the probabilities. Use the value of alpha
        given by `self.alpha`.

        Parameters
        ----------
        y : np.ndarray
            A 1D array of length `n`, with the labels of the `n` examples.
            Note that this array may be corresponding to a subset of all
            examples, so you may not see all possible labels in `y`, but
            you still have to compute the probabilities for all possible
            labels

        Returns
        -------
        probs : dict (str -> float)
            The computed class probabilities. The keys are all the possible
            class labels, the values are the computed probabilities.
        """

        probs = collections.OrderedDict()
        n = len(y)
        counter = collections.Counter(y)
        for label in sorted(self.class_labels):
            numerator = counter[label] + self.alpha
            denominator = n + self.alpha * len(self.class_labels)
            probs[label] = numerator / denominator

        return probs

    def subset_data(self, X, y, j, v):
        """Split a dataset based on the given value of a given attribute.

        This function must return the subset of `X` and `y` for which the `j`-th
        attribute has the value `v`.

        Parameters
        ----------
        X : np.ndarray
            This is the feature matrix, a 2D numpy array where rows are the
            examples (data points) and the columns are the features.
        y : np.ndarray
            A 1D array of length `n`, with the labels of the `n` examples
        j : int
            The integer index that specify which attribute to use
        v : str
            The value of the attribute to use for splitting the data

        Returns
        -------
        X_subset, y_subset : np.ndarray
            The subset of X and y that corresponds to the examples (data points)
            that have value `v` for the j-th attribute
        """

        has_value_v = X[:, j] == v
        
        X_subset = X[has_value_v]
        y_subset = y[has_value_v]

        return X_subset, y_subset

    def split_on_best(self, X, y, exclude=set()):
        """This function selects the best attribute to split, i.e. the attribute
        with the highest score, and then uses it to split the data based on the
        chosen attribute

        Parameters
        ----------
        X : np.ndarray
            This is the feature matrix, a 2D numpy array where rows are the
            examples (data points) and the columns are the features.
        y : np.ndarray
            A 1D array of length `n`, with the labels of the `n` examples
        exclude: set
            Set of column indexes to exclude from consideration

        Returns
        -------
        subsets : dict of str -> (np.ndarray, np.ndarray)
            A list with the subsets of the data, according to each split
        feature_idx: int
            The index (column number) of the feature selected for the split
        """

        num_features = X.shape[1]

        best_gain, best_feature = -float('inf'), None
        for j in range(num_features):
            # Skip the features that we've been told to exclude from
            # consideration (e.g. avoid repeating features in decision trees)
            if j in exclude: continue

            ## >>> YOUR CODE HERE >>>
            gain = self.compute_gain(X, y, j)
            if gain > best_gain:
                best_gain = gain
                best_feature = j
            ## <<< END OF YOUR CODE <<<


        # This creates a dictionary that has a default value, i.e.,
        # if you try to get the value for a key that doesn't exist, instead of
        # giving an error, it returns a tuple of empty lists
        subsets = collections.defaultdict(lambda: (np.empty(0), np.empty(0)))

        # Populate the `subsets` variable with the subsets of the dataset
        # corresponding to each of the splits of the best feature
        ## >>> YOUR CODE HERE >>>
        if best_feature is not None:
            values = np.unique(X[:, best_feature])
            for value in values:
                X_subset, y_subset = self.subset_data(X, y, best_feature, value)
                subsets[value] = (X_subset, y_subset)
        ## <<< END OF YOUR CODE <<<

        return subsets, best_feature

    def __repr__(self):
        return f"{self.__class__.__name__}"


class InformationGain(UncertaintyScorer):
    """Implements a scorer that computes Information Gain (based on entropy)"""

    def score(self, y):
        """Compute the entropy for a set of examples.

        Parameters
        ----------
        y : np.ndarray
            A 1D array of length `n`, with the labels of the `n` examples

        Returns
        -------
        entropy : float
            The computed entropy
        """
        probs = self.compute_class_probs(y)

        ## >>> YOUR CODE HERE >>>
        x = probs.values()
        entropy = -np.sum([p * np.log2(p) for p in x if p > 0])
        ## <<< END OF YOUR CODE <<<

        return entropy

    def compute_gain(self, X, y, j):
        """Compute entropy for the given dataset and attribute

        Parameters
        ----------
        X : np.ndarray
            This is the feature matrix, a 2D numpy array where rows are the
            examples (data points) and the columns are the features.
        y : np.ndarray
            A 1D array of length `n`, with the labels of the `n` examples
        j : int
            The integer index that specify which attribute to use

        Returns
        -------
        gain : float
            The computed entropy for the dataset (X, y) and the j-th attribute
            of X.
        """

        ## >>> YOUR CODE HERE >>>
        first_entropy = self.score(y)
        values = np.unique(X[:,j])
        weighted_entropy = 0
        for v in values:
            X_subset, y_subset = self.subset_data(X, y, j, v)
            prob_v = len(y_subset) / len(y)
            weighted_entropy += prob_v * self.score(y_subset)
        information_gain = first_entropy - weighted_entropy
        ## <<< END OF YOUR CODE <<<

        return information_gain


