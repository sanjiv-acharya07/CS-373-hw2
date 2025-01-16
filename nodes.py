from abc import ABC, abstractmethod
import numpy as np

class TreeNode(ABC):
    """This class describes the interface of the nodes in a tree. It will be
    implemented as either a leaf node or a split node. Split nodes are
    intermediary nodes along the tree, that decide which branch of the tree
    should be used for the prediction of the examples. Leaf nodes are the final
    nodes of the tree, and whenever we reach a leaf node, we are ready to make
    a prediction.
    """

    @abstractmethod
    def predict_proba(self, x):
        """Give the predicted class probabilities for the examples in `x`.

        Arguments
        ---------
        x: np.ndarray
            A 2D array of shape (num_examples, num_features), representing the
            features of the examples that reached this node, for which we
            are making predictions

        Returns
        -------
        probs: np.ndarray
            A 2D array of shape (num_examples, num_classes) with the predicted
            probabilities for each example, for each class.
            Each row has the predictions for the corresponding example in the
            input `x` (in the same order). The columns correspond to the
            possible classes, ordered alpha-numerically (i.e. sorted according
            to python's `sort()` function).
        """
        ...

    @abstractmethod
    def predict(self, x):
        """This function returns the predicted label for the examples that
        reach this node.

        Arguments
        ---------
        x: np.ndarray
            A 2D array of shape (num_examples, num_features), representing the
            features of the examples that reached this node, for which we
            are making predictions

        Returns
        -------
        predicted_labels: np.ndarray
            A 1D numpy array with the labels (strings) predicted for the given
            examples. The length of the array must be `num_examples`.
        """
        ...

class LeafNode(TreeNode):
    """A leaf node of the decision tree.

    At the leaf node, no further decision needs to be made, it simply needs to
    give as an answer the distribution of labels of the training examples which
    have reached this tree.

    Attributes
    ----------
    probs: OrderedDict
        A dictionary that maps each possible class label to the corresponding
        probability value
    class_labels: np.ndarray
        A 1D array with the (ordered) possible class labels

    """

    def __init__(self, probs):
        """When the node is created, we pass the dictionary with the label
        distribution associated with this label, computed from the training
        data.

        Arguments
        ---------
        probs: OrderedDict
            This is a dictionary that maps the possible class labels to a
            probability value.
        """
        self.probs = probs
        self.class_labels = np.array(sorted(probs.keys()))

    def predict_proba(self, x):
        """When making predictions, we simply need to return the probabilities
        computed for this node, regardless of what the examples that reached
        here look like.

        Arguments
        ---------
        x: np.ndarray
            A 2D array of shape (num_examples, num_features), representing the
            features of the examples that reached this leaf node, for which we
            are making predictions

        Returns
        -------
        probs: np.ndarray
            A 2D array of shape (num_examples, num_classes) with the predicted
            probabilities for each example, for each class.
            Each row is the predictions for the corresponding example in the
            input `x` (in the same order). The columns correspond to the
            possible classes, ordered alpha-numerically (i.e. sorted according
            to python's `sort()` function).
        """
        ## >>> YOUR CODE HERE >>>
        num_examples = x.shape[0]
        class_probs = np.array([self.probs[label] for label in self.class_labels])
        probs = np.tile(class_probs, (num_examples, 1))
        ## <<< END OF YOUR CODE <<<

        return probs

    def predict(self, x):
        """This function returns the predicted label for the examples that
        reach this node.

        Arguments
        ---------
        x: np.ndarray
            A 2D array of shape (num_examples, num_features), representing the
            features of the examples that reached this leaf node, for which we
            are making predictions

        Returns
        -------
        predicted_labels: np.ndarray
            A 1D numpy array with the labels (strings) predicted for the given
            examples. The length of the array must be `num_examples`.
        """
        probs = self.predict_proba(x)

        ## >>> YOUR CODE HERE >>>
        max_probs = np.argmax(probs, axis=1)
        predicted_labels = self.class_labels[max_probs]
        ## <<< END OF YOUR CODE <<<

        return predicted_labels

    def __repr__(self):
        s = "[Leaf Node]\n"
        for label, prob in self.probs.items():
            s += f"└-- Label: {label} :: Probability: {prob * 100:5.2f} %\n"
        return s.strip()

class SplitNode(TreeNode):
    """A Split Node in the the tree. Each split node will direct the prediction
    to one of its children, based on the value of a given feature.

    Attributes
    ----------
    feature: int
        This gives the integer index (which column) of the feature associated
        with this node
    children: dict of str -> TreeNode
        This is a dictionary that maps each possible value of the `feature` to
        the corresponding TreeNode. The number of elements in the dictionary
        is the same as the number of possible values of the feature.

    """
    def __init__(self, feature, children):
        """Build the node, with the given feature an children"""
        self.feature = feature
        self.children = children

    def predict_proba(self, x):
        """Compute the predicted class probabilities for the examples in `x`.
        See `TreeNode` for more detail.
        """
        return self._collect_results_recursively(x, 'predict_proba')

    def predict(self, x):
        """Computes the predicted label for the given examples. See `TreeNode`
        for more details
        """
        return self._collect_results_recursively(x, 'predict')

    def _collect_results_recursively(self, x, func_name):
        """This function will call the function `func_name` in each child node,
        corresponding to each feature value, with the appropriate subset of the
        data in `x`. It then collects and returns the result.
        """
        # Split the data in `x`, getting the indices of the data points that
        # correspond to each of the values of this node's feature
        splits = self._choose_branch(x)

        # We'll collect the results (and the ordering of the data points) with
        # these two lists. This indices in `all_idxs` will be used to reorder
        # the result once we collect them all, to ensure that it follows the
        # same order as the original input `x`
        result = []
        all_idxs = []

        # For each split of the data, recursively call the desired function and
        # collect the results (and indices of the splits)
        for v, idxs in splits.items():
            # If we never saw this value in before, treat it as 'NA'
            if v not in self.children:
                v = 'NA'
            # Here we are looking up the function with name given by `func_name`
            # in the child node. This is what allows us to use the same piece
            # of code with more than one function
            child_node_function = getattr(self.children[v], func_name)

            # Call the function with the relevant subset of the data and store
            # the result (and the indices of the subset of the data used)
            result.append(child_node_function(x[idxs]))
            all_idxs.append(idxs)

        # Concatenate the results obtained from each subset of the data and then
        # reorder it, to match the original input order
        result = np.concatenate(result)
        all_idxs = np.concatenate(all_idxs)
        result = result[np.argsort(all_idxs)]  # Re-order them

        return result

    def _choose_branch(self, x):
        """This function is responsible for computing, for each example in `x`,
        which branch of the tree (which child node) to use.

        Arguments
        ---------
        x: np.ndarray
            The input examples, of shape (num_examples, num_features). See
            `TreeNode` for more details.

        Returns
        -------
        splits: dict of str -> np.ndarray
            The returning dictionary maps each possible value of the feature
            (i.e., each of the children nodes) to a numpy array corresponding
            to the indices of `x` (i.e. which rows/examples in `x`) that should
            use the branch of the tree corresponding to the key value.
        """
        # Get the set of unique values in `x` for the feature associated with
        # this node. Note that in some cases (e.g. when using cross-validation,
        # the training data may not have seen all possible feature values, so
        # there may be values here for which we don't have a children node)
        observed_values = set(x[:, self.feature])

        ## >>> YOUR CODE HERE >>>
        splits = {}
        for val in observed_values:
            splits[val] = np.where(x[:, self.feature] == val)[0]
        ## <<< END OF YOUR CODE <<<

        return splits

    def __repr__(self):
        """Returns a string representation if the node"""
        s = f"[Split Node :: Feature: {self.feature}]\n"
        for i, (k, node) in enumerate(sorted(self.children.items())):
            c = "|" if i != len(self.children) - 1 else " "
            s += f"└-- Feature {self.feature} == {k}\n"
            s += "\n".join([f"{c}   {x}" for x in str(node).split("\n")])
            s += "\n"
        return s.strip()


