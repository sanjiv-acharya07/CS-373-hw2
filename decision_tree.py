from nodes import SplitNode, LeafNode
import pandas as pd

class DecisionTree:
	def __init__(self, scorer, max_depth=5):
		"""
		This class implements a Decision Tree. It uses the given scorer to decide which
		attributes to split at each branch and it limits the depth of the tree to the
		given value.

		Args:
			scorer: which scorer to use for the splits (InformationGain or GiniGain)
			max_depth: maximum depth of the decision tree to build
		"""
		self.scorer = scorer
		self.max_depth = max_depth
		self.root = None  # root of decision tree, assigned after fit()

	def fit(self, X, y):
		"""This function builds the tree, based on the given dataset"""
		self.root = self._build(X, y, self.max_depth)

	def __repr__(self):
		s = f"DecisionTree(scorer={self.scorer}, max_depth={self.max_depth})\n"
		s += str(self.root)
		return s

	def predict(self, x):
		"""Predict the label of a set of data points.

		Arguments
		---------
		x: np.ndarray
			A 2D array of shape (num_examples, num_features), representing the
			features of the data points for which we are making predictions

		Returns
		-------
		predicted_labels: np.ndarray
			A 1D numpy array with the labels (strings) predicted for the given
			examples. The length of the array must be `num_examples`.
		"""
		assert self.root is not None, "must call fit() first"
		return self.root.predict(x)

	def predict_proba(self, x):
		"""Give the predicted class probabilities for the data points in `x`.

		Arguments
		---------
		x: np.ndarray
			A 2D array of shape (num_examples, num_features), representing the
			features of the examples for which we are making predictions

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
		assert self.root is not None, "must call fit() first"
		return self.root.predict_proba(x)

	def _build(self, x, y, max_depth, exclude=set()):
		"""Recursively build the decision tree given data points and maximum
		depth allowed.

		Arguments
		---------
		x: np.ndarray
			A 2D array of shape (num_examples, num_features), with the data
			points to use.
		y: np.ndarray
			A 1D array of shape (num_examples,), with the labels of the data
			points.
		max_depth: int
			the maximum depth allowed for the subtree.
		exclude: set of ints
			Set of column indices to exclude from the consideration for splits
		"""
		assert len(x) == len(y), "x andy must have same length"
		assert len(y) > 0, "you must give some examples"

		# Whenever we reach the maximum allowed depth or have a pure dataset,
		# we stop growing the tree and create a leaf node
		if max_depth <= 0 or self._is_pure(y) or len(exclude) == x.shape[1]:
			return LeafNode(self.scorer.compute_class_probs(y))

		splits, feature_of_split = self.scorer.split_on_best(x, y, exclude)

		children = {}
		## >>> YOUR CODE HERE >>>
		for value, (x_subset, y_subset) in splits.items():
			if len(y_subset) == 0:
				children[value] = LeafNode(self.scorer.compute_classprobs(y))
			else:
				children[value] = self._build(x_subset, y_subset, max_depth - 1, exclude)
		## <<< END OF YOUR CODE <<<

		# Make sure to always have an 'NA' node in the split (even if we did
		# not see any 'NA' in the training data). This will allow us to give
		# treat new feature values (seen in test but not in training) as 'NA'
		if 'NA' not in children:
			## >>> YOUR CODE HERE >>>
			children['NA'] = LeafNode(self.scorer.compute_class_probs(y))
	 		## <<< END OF YOUR CODE <<<

		return SplitNode(feature_of_split, children)

	def _is_pure(self, y):
		"""Check if the set of labels in `y` is pure, i.e., all labels are
		the same.
		"""
		return len(set(y)) == 1



"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from gain import *
from data_split import *


def load_cars_data(path: str):
    """
    Load the csv file and return both features and labels.
    The column "class" is the label column, the possible values are "unacc" (unacceptable), "acc" (acceptable), "good", "vgood" (very good).
    All other columns are features columns. Their categorical values are the following:
	buying: vhigh, high, med, low.
    maint: vhigh, high, med, low.
    doors: 2, 3, 4, 5more.
    persons: 2, 4, more.
    lug_boot: small, med, big.
    safety: low, med, high.

    Return:
            features (numpy 2d array), labels (numpy 1d array)
    """
	
    column_names=['buying','maint','doors','persons','lug_boot','safety','class']
    data = pd.read_csv(os.path.join(os.path.dirname(
        __file__), path),names=column_names).fillna("NA").astype(str)

    # The last column is the label (class)
    y = data.iloc[:, data.shape[1]-1].values

    # The remaining columns are the features
    X = data.iloc[:, :-1].values
    
    return X, y

def zero_one_loss(y_true, y_pred):
	"""Computes zero-one loss

	Arguments
	---------
	y_true: np.ndarray
		A 1D numpy array with the true labels
	y_pred: np.ndarray
		A 1D numpy array with the predicted labels

	Returns
	-------
	loss: float
		The compute zero-one loss for the given data points
	"""
	assert len(y_true) == len(y_pred), "Input arrays must have same length"

	return (y_true != y_pred).astype(float).mean()

def accuracy(original, predictions):
    """
    Calculate the accuracy of given predictions on the given labels.

    Args:
        original: The original labels of shape (N,).
        predictions: Predictions of shape (N,).

    Returns:
        accuracy: The accuracy of the predictions.
    """
    return np.mean(original == predictions)

def decision_tree_performance(model, X, y, loss_fn=zero_one_loss):
	"""This function will learn a Decision Tree with the given scorer and depth based
	given dataset `X` and `y`, and compute the loss and accuracy.
	"""
	y_pred = model.predict(X)

	loss = loss_fn(y, y_pred)
	acc = accuracy(y, y_pred)
	return loss, acc

def evaluate_DecisionTrees():
	"""
	Evaluate the Decision Trees performance on the data set.
	This function is supposed to do two things:
	1) Print the DT model with max_depth=1, along with the loss and accuracy.
	2) Plot the loss and accuracy figures vs. max_depth

	PLEASE DO NOT CHANGE!
	"""
	X, y = load_cars_data(
		path.join(path.dirname(__file__), "dataset/car_evaluation.csv")
	)

	# Split the dataset into a training set and a validation set
	X_train, X_val, y_train, y_val = my_train_valid_split(X, y)


	# Print the DT model with max_depth=1, along with the loss and accuracy.
	scorer = InformationGain(class_labels=set(y))
	model = DecisionTree(scorer, max_depth=1)
	model.fit(X, y)

	loss, acc = decision_tree_performance(model, X, y)
	print(model)
	print("0-1 Loss:", loss)
	print("Accuracy: ", acc)
	print('\n------------------------------------------\n')


	## Plot the loss and accuracy figures vs. max_depth
	max_depths = list(range(1, 12)) 
	train_losses, val_losses = [], []
	train_accs, val_accs = [], []
	for depth in range(1, 12):
		model = DecisionTree(scorer, max_depth=depth)
		model.fit(X_train, y_train)
		train_loss, train_acc = decision_tree_performance(model, X_train, y_train)
		val_loss, val_acc = decision_tree_performance(model, X_val, y_val)

		train_losses.append(train_loss)
		val_losses.append(val_loss)

		train_accs.append(train_acc)
		val_accs.append(val_acc)

	# plot of losses
	fig1 = plt.figure()
	plt.plot(max_depths, train_losses, '-bo', label='training')
	plt.plot(max_depths, val_losses, '-ro', label='validation')
	plt.legend()
	plt.xlabel('max_depth')
	plt.ylabel('0-1 loss')
	plt.grid(True)
	fig1.savefig(os.path.join(os.path.dirname(__file__), "dt_losses.png"))

	# plot of accuracies
	fig2 = plt.figure()
	plt.plot(max_depths, train_accs, '-bo', label='training')
	plt.plot(max_depths, val_accs, '-ro', label='validation')
	plt.legend()
	plt.xlabel('max_depth')
	plt.ylabel('accuracy')
	plt.grid(True)
	fig2.savefig(os.path.join(os.path.dirname(__file__), 'dt_accs.png'))




if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    evaluate_DecisionTrees()
