{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CSE 4334/5334 Programming Assignment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Fall 2017"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Due: 11:59pm Central Time, Friday, December 8, 2017"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The instructions on this assignment are written in an .ipynb file. You can use the following commands to install the Jupyter notebook viewer. \"pip\" is a command for installing Python packages. You are required to use Python 3.5.1 or above in this project. \n",
    "\n",
    "    pip install jupyter\n",
    "\n",
    "    pip install notebook (You might have to use \"sudo\" if you are installing them at system level)\n",
    "\n",
    "To run the Jupyter notebook viewer, use the following command:\n",
    "\n",
    "    jupyter notebook P1.ipynb\n",
    "\n",
    "The above command will start a webservice at http://localhost:8888/ and display the instructions in the '.ipynb' file.\n",
    "\n",
    "The same instructions are also available at https://nbviewer.jupyter.org/url/crystal.uta.edu/~cli/cse5334/ipythonnotebook/P1.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Requirements"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* This assignment must be done individually. You must implement the whole assignment by yourself. Academic dishonety will have serious consequences.\n",
    "* You can discuss topics related to the assignment with your fellow students. But you are not allowed to discuss/share your solution and code."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Introduction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "Part of the materials in this instruction notebook are adapted from \"Introduction to Machine Learning with Python\" by Andreas C. Mueller and Sarah Guido.\n",
    "\n",
    "To run the examples in this notebook and to finish your assignment, you need a few Python modules. If you already have a Python installation set up, you can use pip to install all of these packages:\n",
    "\n",
    "$ pip install numpy matplotlib ipython jupyter scikit-learn pandas graphviz\n",
    "\n",
    "In your python code, you will always need to import a subset of the following modules. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import graphviz\n",
    "\n",
    "\n",
    "from IPython.display import display\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.tree import export_graphviz"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## The Breast Cancer Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The first dataset that we use in this notebook is included in scikit-learn, a popular machine learning library for Python. The dataset is the Wisconsin Breast Cancer dataset, which records clinical measurements of breast cancer tumors. Each tumor is labeled as “benign” (for harmless tumors) or “malignant” (for cancerous tumors), and the task is to learn to predict whether a tumor is malignant based on the measurements of the tissue.\n",
    "\n",
    "The data can be loaded using the load_breast_cancer function from scikit-learn:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cancer.keys(): dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])\n"
     ]
    }
   ],
   "source": [
    "cancer = load_breast_cancer()\n",
    "print(\"cancer.keys(): {}\".format(cancer.keys()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Datasets that are included in scikit-learn are usually stored as Bunch objects, which contain some information about the dataset as well as the actual data. All you need to know about Bunch objects is that they behave like dictionaries, with the added benefit that you can access values using a dot (as in bunch.key instead of bunch['key'])."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The dataset consists of 569 data points, with 30 features each:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of cancer data: (569, 30)\n"
     ]
    }
   ],
   "source": [
    "print(\"Shape of cancer data: {}\".format(cancer.data.shape))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Of these 569 data points, 212 are labeled as malignant and 357 as benign:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sample counts per class:\n",
      "{'malignant': 212, 'benign': 357}\n"
     ]
    }
   ],
   "source": [
    "print(\"Sample counts per class:\\n{}\".format(\n",
    "      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To get a description of the semantic meaning of each feature, we can have a look at the feature_names attribute:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature names:\n",
      "['mean radius' 'mean texture' 'mean perimeter' 'mean area'\n",
      " 'mean smoothness' 'mean compactness' 'mean concavity'\n",
      " 'mean concave points' 'mean symmetry' 'mean fractal dimension'\n",
      " 'radius error' 'texture error' 'perimeter error' 'area error'\n",
      " 'smoothness error' 'compactness error' 'concavity error'\n",
      " 'concave points error' 'symmetry error' 'fractal dimension error'\n",
      " 'worst radius' 'worst texture' 'worst perimeter' 'worst area'\n",
      " 'worst smoothness' 'worst compactness' 'worst concavity'\n",
      " 'worst concave points' 'worst symmetry' 'worst fractal dimension']\n"
     ]
    }
   ],
   "source": [
    "print(\"Feature names:\\n{}\".format(cancer.feature_names))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's print out the names of the features (attributes) and the values in the target (class attribute), and the first 3 instances in the dataset. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['mean radius' 'mean texture' 'mean perimeter' 'mean area'\n",
      " 'mean smoothness' 'mean compactness' 'mean concavity'\n",
      " 'mean concave points' 'mean symmetry' 'mean fractal dimension'\n",
      " 'radius error' 'texture error' 'perimeter error' 'area error'\n",
      " 'smoothness error' 'compactness error' 'concavity error'\n",
      " 'concave points error' 'symmetry error' 'fractal dimension error'\n",
      " 'worst radius' 'worst texture' 'worst perimeter' 'worst area'\n",
      " 'worst smoothness' 'worst compactness' 'worst concavity'\n",
      " 'worst concave points' 'worst symmetry' 'worst fractal dimension'] ['malignant' 'benign']\n",
      "[  1.79900000e+01   1.03800000e+01   1.22800000e+02   1.00100000e+03\n",
      "   1.18400000e-01   2.77600000e-01   3.00100000e-01   1.47100000e-01\n",
      "   2.41900000e-01   7.87100000e-02   1.09500000e+00   9.05300000e-01\n",
      "   8.58900000e+00   1.53400000e+02   6.39900000e-03   4.90400000e-02\n",
      "   5.37300000e-02   1.58700000e-02   3.00300000e-02   6.19300000e-03\n",
      "   2.53800000e+01   1.73300000e+01   1.84600000e+02   2.01900000e+03\n",
      "   1.62200000e-01   6.65600000e-01   7.11900000e-01   2.65400000e-01\n",
      "   4.60100000e-01   1.18900000e-01] 0\n",
      "[  2.05700000e+01   1.77700000e+01   1.32900000e+02   1.32600000e+03\n",
      "   8.47400000e-02   7.86400000e-02   8.69000000e-02   7.01700000e-02\n",
      "   1.81200000e-01   5.66700000e-02   5.43500000e-01   7.33900000e-01\n",
      "   3.39800000e+00   7.40800000e+01   5.22500000e-03   1.30800000e-02\n",
      "   1.86000000e-02   1.34000000e-02   1.38900000e-02   3.53200000e-03\n",
      "   2.49900000e+01   2.34100000e+01   1.58800000e+02   1.95600000e+03\n",
      "   1.23800000e-01   1.86600000e-01   2.41600000e-01   1.86000000e-01\n",
      "   2.75000000e-01   8.90200000e-02] 0\n",
      "[  1.96900000e+01   2.12500000e+01   1.30000000e+02   1.20300000e+03\n",
      "   1.09600000e-01   1.59900000e-01   1.97400000e-01   1.27900000e-01\n",
      "   2.06900000e-01   5.99900000e-02   7.45600000e-01   7.86900000e-01\n",
      "   4.58500000e+00   9.40300000e+01   6.15000000e-03   4.00600000e-02\n",
      "   3.83200000e-02   2.05800000e-02   2.25000000e-02   4.57100000e-03\n",
      "   2.35700000e+01   2.55300000e+01   1.52500000e+02   1.70900000e+03\n",
      "   1.44400000e-01   4.24500000e-01   4.50400000e-01   2.43000000e-01\n",
      "   3.61300000e-01   8.75800000e-02] 0\n"
     ]
    }
   ],
   "source": [
    "print(cancer.feature_names,cancer.target_names)\n",
    "for i in range(0,3):\n",
    "    print(cancer.data[i], cancer.target[i])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can find out more about the data by reading cancer.DESCR if you are interested."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## k-Nearest Neighbor\n",
    "#### k-Neighbors Classification"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now let’s look at how we can apply the k-nearest neighbors algorithm using scikit-learn. First, we split our data into a training and a test set so we can evaluate generalization performance: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_feature, test_feature, train_class, test_class = train_test_split(\n",
    "    cancer.data, cancer.target, stratify=cancer.target, random_state=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note that this function randomly partitions the dataset into training and test sets. The randomness is controlled by a pseudo random number generator, which generates random numbers using a seed. If you fix the seed, you will actually always get the same partition (thus no randomness). That is why we set random_state=0. (We can also use any other fixed number instead of 0, to acheive the same effect.) It guarantees that you reproduce the same results in every run. It is useful in testing your programs. However, in your real production code where randomness is needed, you shouldn't fix random_state. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we instantiate the KNeighborsClassifier class. This is when we can set parameters, like the number of neighbors to use. Here, we set it to 3:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "knn = KNeighborsClassifier(n_neighbors=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, we fit the classifier using the training set. For KNeighborsClassifier this means storing the dataset, so we can compute neighbors during prediction:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "           metric_params=None, n_jobs=1, n_neighbors=3, p=2,\n",
       "           weights='uniform')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn.fit(train_feature, train_class)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To make predictions on the test data, we call the predict method. For each data point in the test set, this computes its nearest neighbors in the training set and finds the most common class among these:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set predictions:\n",
      "[0 0 0 1 0 1 0 0 0 1 0 0 1 1 1 1 1 0 1 0 0 0 0 1 1 0 1 1 1 0 1 1 0 1 1 1 0\n",
      " 0 0 1 0 0 0 0 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 1 1 0 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 0 0 1 1 1 0 0 1 0 1 1 1 0 0 1 1 1\n",
      " 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 0]\n"
     ]
    }
   ],
   "source": [
    "print(\"Test set predictions:\\n{}\".format(knn.predict(test_feature)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To evaluate how well our model generalizes, we can call the score method with the test data together with the test labels:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set accuracy: 0.92\n"
     ]
    }
   ],
   "source": [
    "print(\"Test set accuracy: {:.2f}\".format(knn.score(test_feature, test_class)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We see that our model is about 92% accurate, meaning the model predicted the class correctly for 92% of the samples in the test dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Analyzing KNeighborsClassifier"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let’s investigate whether we can confirm the connection between model complexity and generalization. For that, we evaluate training and test set performance with different numbers of neighbors. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nnFpLkZGR/P3vf2fRokW0bt36pGvffvvtPPDAAwBcc801fPLJJ5x//vlcddVVTJs2jYsuuoi8vDwK\nCwvLLRteleXLl7Ny5UpiYmLIz8/nww8/pGXLluzdu5chQ4YwceJE1q5dy5/+9Ce++eYbWrduzf79\n+4mIiGD48OF8+umnXHjhhbz99ttcfPHFhISE1OQTrjPWwvDSwM4xjO3Vlhe+3Mz+w8fdDseYBmv+\n/PnMnz+f/v37M2DAANLS0ti4cSN9+vTh888/55577uGrr74iMjKyymstWrSI008/nT59+rBw4ULW\nrFnDwYMH2b59OxdddBHgFO0LDw+vsGx4ZcaMGVN8nKpy33330bdvX0aPHs327dvZvXs3Cxcu5LLL\nLitOaGXLkQN+W7/C16yFUQ2/H9+dsf9YwtMLNwZMGWhjfKqSlkBdUVXuvfdebr755pP2LV++nDlz\n5nD//fczatSo4tZDefLy8rj11ltJSUmhY8eOPPTQQ34tR/7WW2+RlZVFamoqISEhxMfHV3q/oUOH\nkpGRweLFiykoKCAxMbHasdU1a2FUw2ltIvj1oI7867stbN1nszKM8YWy5c3HjRvHK6+8wqFDhwDY\nvn07e/bsYceOHYSHh3P11Vdz9913s3z58nLPL1L0Zd26dWsOHTpUvExrREQEcXFxfPTRR4BTePDI\nkSMVlg0vWY686BrlycnJoU2bNoSEhLBo0SK2bNkCwMiRI3n33XfZt29fqesCXHvttUyaNKletC7A\nEka13Tm6G0FNhP/73EqGGOMLZcubjx07lkmTJnHGGWfQp08fLr30Ug4ePMiqVasYPHgw/fr14+GH\nH+b+++8HYPLkyYwfP/6kQe+oqChuuukmEhMTGTduXPGKeABvvvkmTz31FH379uXMM89k165dFZYN\nv+uuu3j++efp379/uet8F7nqqqtISUmhT58+vPHGG/To0QOA3r1784c//IFzzjmHpKQkfve735U6\nJzs7myuvrHS9uYBh5c1r4G/z0nh20SY+mTqMxA5V96MaE8isvLl73nvvPWbPns2bb75ZJ/erbXlz\na2HUwM3ndCE6PITpn6W5HYoxpp6aOnUq06ZN449//KPboXjNEkYNtAwN4faRXVmavpevNma5HY4x\nph56+umnSU9Pp1u3bm6H4jVLGDV09ZBOxEWHMf2zNAoLG063nmmcGlLXtCmfL/4dW8KooWbBQdw9\n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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11d04e518>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "cancer = load_breast_cancer()\n",
    "train_feature, test_feature, train_class, test_class = train_test_split(\n",
    "    cancer.data, cancer.target, stratify=cancer.target, random_state=0)\n",
    "\n",
    "training_accuracy = []\n",
    "test_accuracy = []\n",
    "# try n_neighbors from 1 to 10.\n",
    "neighbors_settings = range(1, 11)\n",
    "\n",
    "for n_neighbors in neighbors_settings:\n",
    "    # build the model\n",
    "    knn = KNeighborsClassifier(n_neighbors=n_neighbors)\n",
    "    knn.fit(train_feature, train_class)\n",
    "    # record training set accuracy\n",
    "    training_accuracy.append(knn.score(train_feature, train_class))\n",
    "    # record generalization accuracy\n",
    "    test_accuracy.append(knn.score(test_feature, test_class))\n",
    "    \n",
    "plt.plot(neighbors_settings, training_accuracy, label=\"training accuracy\")\n",
    "plt.plot(neighbors_settings, test_accuracy, label=\"test accuracy\")\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"n_neighbors\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The plot shows the training and test set accuracy on the y-axis against the setting of n_neighbors on the x-axis. While real-world plots are rarely very smooth, we can still recognize some of the characteristics of overfitting and underfitting. Considering a single nearest neighbor, the prediction on the training set is perfect. But when more neighbors are considered, the model becomes simpler and the training accuracy drops. The test set accuracy for using a single neighbor is lower than when using more neighbors, indicating that using the single nearest neighbor leads to a model that is too complex. On the other hand, when considering 10 neighbors, the model is too simple and performance is even worse. (It is not a typo. Yes, using less neighbors leads to more complex models. Think carefully about this.) The best performance is somewhere in the middle, using around six neighbors. Still, it is good to keep the scale of the plot in mind. The worst performance is around 88% accuracy, which might still be acceptable."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Linear Support Vector Machines"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Linear support vector machines (linear SVMs) is implemented in svm.LinearSVC. Let's apply it on the brest cancer dataset. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set score: 0.825\n"
     ]
    }
   ],
   "source": [
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import LinearSVC\n",
    "\n",
    "cancer = load_breast_cancer()\n",
    "train_feature, test_feature, train_class, test_class = train_test_split(\n",
    "    cancer.data, cancer.target, stratify=cancer.target, random_state=0)\n",
    "\n",
    "linearsvm = LinearSVC(random_state=0).fit(train_feature, train_class)\n",
    "print(\"Test set score: {:.3f}\".format(linearsvm.score(test_feature, test_class)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Naive Bayes Classifiers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Naive Bayes classifiers are also implemented in scikit-learn. Since the features in the breast cancer dataset are all continuous numeric attributes, let's use GaussianNB. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set score: 0.923\n"
     ]
    }
   ],
   "source": [
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "cancer = load_breast_cancer()\n",
    "train_feature, test_feature, train_class, test_class = train_test_split(\n",
    "    cancer.data, cancer.target, stratify=cancer.target, random_state=0)\n",
    "\n",
    "nb = GaussianNB().fit(train_feature, train_class)\n",
    "print(\"Test set score: {:.3f}\".format(nb.score(test_feature, test_class)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Decision trees"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Decision trees are also implmented in scikit-learn. Let's use DecisionTreeClassifier. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "uuid": "6e5d7a76-9bba-42f7-b26e-907775d289b2"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set score: 1.000\n",
      "Test set score: 0.937\n"
     ]
    }
   ],
   "source": [
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "cancer = load_breast_cancer()\n",
    "train_feature, test_feature, train_class, test_class = train_test_split(\n",
    "    cancer.data, cancer.target, stratify=cancer.target, random_state=42)\n",
    "\n",
    "tree = DecisionTreeClassifier(random_state=0)\n",
    "tree.fit(train_feature, train_class)\n",
    "print(\"Training set score: {:.3f}\".format(tree.score(train_feature, train_class)))\n",
    "print(\"Test set score: {:.3f}\".format(tree.score(test_feature, test_class)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "If we don’t restrict the depth of a decision tree, the tree can become arbitrarily deep and complex. Unpruned trees are therefore prone to overfitting and not generalizing well to new data. Now let’s apply pre-pruning to the tree, which will stop developing the tree before we perfectly fit to the training data. One option is to stop building the tree after a certain depth has been reached. In the above code, we didn't set max_depth (i.e., max_depth= None,  which is the default value). Nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split instances (min_samples_split is another parameter in DecisionTreeClassifier). Now let's set max_depth=4, meaning only four consecutive questions can be asked. Limiting the depth of the tree decreases overfitting. This leads to a lower accuracy on the training set, but an improvement on the test set:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set score: 0.988\n",
      "Test set score: 0.951\n"
     ]
    }
   ],
   "source": [
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "cancer = load_breast_cancer()\n",
    "train_feature, test_feature, train_class, test_class = train_test_split(\n",
    "    cancer.data, cancer.target, stratify=cancer.target, random_state=42)\n",
    "\n",
    "tree = DecisionTreeClassifier(max_depth=4, random_state=0)\n",
    "tree.fit(train_feature, train_class)\n",
    "print(\"Training set score: {:.3f}\".format(tree.score(train_feature, train_class)))\n",
    "print(\"Test set score: {:.3f}\".format(tree.score(test_feature, test_class)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Analyzing Decision Trees"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can visualize the tree using the export_graphviz function from the tree module. This writes a file in the .dot file format, which is a text file format for storing graphs. We set an option to color the nodes to reflect the majority class in each node and pass the class and features names so the tree can be properly labeled:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.tree import export_graphviz\n",
    "export_graphviz(tree, out_file=\"tree.dot\", class_names=[\"malignant\", \"benign\"],\n",
    "                feature_names=cancer.feature_names, impurity=False, filled=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/svg+xml": [
       "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n",
       "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n",
       " \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n",
       "<!-- Generated by graphviz version 2.38.0 (20140413.2041)\n",
       " -->\n",
       "<!-- Title: Tree Pages: 1 -->\n",
       "<svg width=\"1035pt\" height=\"458pt\"\n",
       " viewBox=\"0.00 0.00 1035.15 458.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n",
       "<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 454)\">\n",
       "<title>Tree</title>\n",
       "<polygon fill=\"white\" stroke=\"none\" points=\"-4,4 -4,-454 1031.15,-454 1031.15,4 -4,4\"/>\n",
       "<!-- 0 -->\n",
       "<g id=\"node1\" class=\"node\"><title>0</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.403922\" stroke=\"black\" points=\"637.222,-450 490.991,-450 490.991,-386 637.222,-386 637.222,-450\"/>\n",
       "<text text-anchor=\"middle\" x=\"564.106\" y=\"-434.8\" font-family=\"Times,serif\" font-size=\"14.00\">worst radius &lt;= 16.795</text>\n",
       "<text text-anchor=\"middle\" x=\"564.106\" y=\"-420.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 426</text>\n",
       "<text text-anchor=\"middle\" x=\"564.106\" y=\"-406.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [159, 267]</text>\n",
       "<text text-anchor=\"middle\" x=\"564.106\" y=\"-392.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 1 -->\n",
       "<g id=\"node2\" class=\"node\"><title>1</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.901961\" stroke=\"black\" points=\"564.091,-350 368.121,-350 368.121,-286 564.091,-286 564.091,-350\"/>\n",
       "<text text-anchor=\"middle\" x=\"466.106\" y=\"-334.8\" font-family=\"Times,serif\" font-size=\"14.00\">worst concave points &lt;= 0.1359</text>\n",
       "<text text-anchor=\"middle\" x=\"466.106\" y=\"-320.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 284</text>\n",
       "<text text-anchor=\"middle\" x=\"466.106\" y=\"-306.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [25, 259]</text>\n",
       "<text text-anchor=\"middle\" x=\"466.106\" y=\"-292.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 0&#45;&gt;1 -->\n",
       "<g id=\"edge1\" class=\"edge\"><title>0&#45;&gt;1</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M533.099,-385.992C523.983,-376.876 513.902,-366.796 504.388,-357.282\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"506.795,-354.739 497.249,-350.142 501.845,-359.688 506.795,-354.739\"/>\n",
       "<text text-anchor=\"middle\" x=\"497.249\" y=\"-370.942\" font-family=\"Times,serif\" font-size=\"14.00\">True</text>\n",
       "</g>\n",
       "<!-- 14 -->\n",
       "<g id=\"node15\" class=\"node\"><title>14</title>\n",
       "<polygon fill=\"#e58139\" fill-opacity=\"0.941176\" stroke=\"black\" points=\"736.467,-350 589.746,-350 589.746,-286 736.467,-286 736.467,-350\"/>\n",
       "<text text-anchor=\"middle\" x=\"663.106\" y=\"-334.8\" font-family=\"Times,serif\" font-size=\"14.00\">texture error &lt;= 0.4732</text>\n",
       "<text text-anchor=\"middle\" x=\"663.106\" y=\"-320.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 142</text>\n",
       "<text text-anchor=\"middle\" x=\"663.106\" y=\"-306.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [134, 8]</text>\n",
       "<text text-anchor=\"middle\" x=\"663.106\" y=\"-292.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 0&#45;&gt;14 -->\n",
       "<g id=\"edge14\" class=\"edge\"><title>0&#45;&gt;14</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M595.431,-385.992C604.64,-376.876 614.823,-366.796 624.434,-357.282\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"627.002,-359.665 631.646,-350.142 622.077,-354.69 627.002,-359.665\"/>\n",
       "<text text-anchor=\"middle\" x=\"631.519\" y=\"-370.942\" font-family=\"Times,serif\" font-size=\"14.00\">False</text>\n",
       "</g>\n",
       "<!-- 2 -->\n",
       "<g id=\"node3\" class=\"node\"><title>2</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.984314\" stroke=\"black\" points=\"358.312,-250 215.901,-250 215.901,-186 358.312,-186 358.312,-250\"/>\n",
       "<text text-anchor=\"middle\" x=\"287.106\" y=\"-234.8\" font-family=\"Times,serif\" font-size=\"14.00\">radius error &lt;= 1.0475</text>\n",
       "<text text-anchor=\"middle\" x=\"287.106\" y=\"-220.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 252</text>\n",
       "<text text-anchor=\"middle\" x=\"287.106\" y=\"-206.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [4, 248]</text>\n",
       "<text text-anchor=\"middle\" x=\"287.106\" y=\"-192.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 1&#45;&gt;2 -->\n",
       "<g id=\"edge2\" class=\"edge\"><title>1&#45;&gt;2</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M409.47,-285.992C391.513,-276.161 371.508,-265.208 352.961,-255.055\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"354.441,-251.875 343.989,-250.142 351.08,-258.015 354.441,-251.875\"/>\n",
       "</g>\n",
       "<!-- 7 -->\n",
       "<g id=\"node8\" class=\"node\"><title>7</title>\n",
       "<polygon fill=\"#e58139\" fill-opacity=\"0.474510\" stroke=\"black\" points=\"537.877,-250 394.336,-250 394.336,-186 537.877,-186 537.877,-250\"/>\n",
       "<text text-anchor=\"middle\" x=\"466.106\" y=\"-234.8\" font-family=\"Times,serif\" font-size=\"14.00\">worst texture &lt;= 25.62</text>\n",
       "<text text-anchor=\"middle\" x=\"466.106\" y=\"-220.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 32</text>\n",
       "<text text-anchor=\"middle\" x=\"466.106\" y=\"-206.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [21, 11]</text>\n",
       "<text text-anchor=\"middle\" x=\"466.106\" y=\"-192.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 1&#45;&gt;7 -->\n",
       "<g id=\"edge7\" class=\"edge\"><title>1&#45;&gt;7</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M466.106,-285.992C466.106,-277.859 466.106,-268.959 466.106,-260.378\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"469.607,-260.142 466.106,-250.142 462.607,-260.142 469.607,-260.142\"/>\n",
       "</g>\n",
       "<!-- 3 -->\n",
       "<g id=\"node4\" class=\"node\"><title>3</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.988235\" stroke=\"black\" points=\"211.936,-150 38.277,-150 38.277,-86 211.936,-86 211.936,-150\"/>\n",
       "<text text-anchor=\"middle\" x=\"125.106\" y=\"-134.8\" font-family=\"Times,serif\" font-size=\"14.00\">smoothness error &lt;= 0.0033</text>\n",
       "<text text-anchor=\"middle\" x=\"125.106\" y=\"-120.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 251</text>\n",
       "<text text-anchor=\"middle\" x=\"125.106\" y=\"-106.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [3, 248]</text>\n",
       "<text text-anchor=\"middle\" x=\"125.106\" y=\"-92.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 2&#45;&gt;3 -->\n",
       "<g id=\"edge3\" class=\"edge\"><title>2&#45;&gt;3</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M235.849,-185.992C219.745,-176.251 201.821,-165.408 185.165,-155.332\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"186.955,-152.324 176.587,-150.142 183.331,-158.313 186.955,-152.324\"/>\n",
       "</g>\n",
       "<!-- 6 -->\n",
       "<g id=\"node7\" class=\"node\"><title>6</title>\n",
       "<polygon fill=\"#e58139\" stroke=\"black\" points=\"344.202,-143 230.011,-143 230.011,-93 344.202,-93 344.202,-143\"/>\n",
       "<text text-anchor=\"middle\" x=\"287.106\" y=\"-127.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 1</text>\n",
       "<text text-anchor=\"middle\" x=\"287.106\" y=\"-113.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [1, 0]</text>\n",
       "<text text-anchor=\"middle\" x=\"287.106\" y=\"-99.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 2&#45;&gt;6 -->\n",
       "<g id=\"edge6\" class=\"edge\"><title>2&#45;&gt;6</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M287.106,-185.992C287.106,-175.646 287.106,-164.057 287.106,-153.465\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"290.607,-153.288 287.106,-143.288 283.607,-153.288 290.607,-153.288\"/>\n",
       "</g>\n",
       "<!-- 4 -->\n",
       "<g id=\"node5\" class=\"node\"><title>4</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.666667\" stroke=\"black\" points=\"96.3196,-50 -0.106681,-50 -0.106681,-0 96.3196,-0 96.3196,-50\"/>\n",
       "<text text-anchor=\"middle\" x=\"48.1064\" y=\"-34.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 4</text>\n",
       "<text text-anchor=\"middle\" x=\"48.1064\" y=\"-20.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [1, 3]</text>\n",
       "<text text-anchor=\"middle\" x=\"48.1064\" y=\"-6.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 3&#45;&gt;4 -->\n",
       "<g id=\"edge4\" class=\"edge\"><title>3&#45;&gt;4</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M98.8228,-85.9375C91.14,-76.8578 82.7332,-66.9225 75.0124,-57.798\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"77.5709,-55.4032 68.4396,-50.0301 72.2272,-59.9248 77.5709,-55.4032\"/>\n",
       "</g>\n",
       "<!-- 5 -->\n",
       "<g id=\"node6\" class=\"node\"><title>5</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.992157\" stroke=\"black\" points=\"219.645,-50 114.568,-50 114.568,-0 219.645,-0 219.645,-50\"/>\n",
       "<text text-anchor=\"middle\" x=\"167.106\" y=\"-34.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 247</text>\n",
       "<text text-anchor=\"middle\" x=\"167.106\" y=\"-20.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [2, 245]</text>\n",
       "<text text-anchor=\"middle\" x=\"167.106\" y=\"-6.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 3&#45;&gt;5 -->\n",
       "<g id=\"edge5\" class=\"edge\"><title>3&#45;&gt;5</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M139.443,-85.9375C143.422,-77.3164 147.757,-67.9239 151.79,-59.1865\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"155.003,-60.5764 156.016,-50.0301 148.647,-57.643 155.003,-60.5764\"/>\n",
       "</g>\n",
       "<!-- 8 -->\n",
       "<g id=\"node9\" class=\"node\"><title>8</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.666667\" stroke=\"black\" points=\"539.846,-150 362.367,-150 362.367,-86 539.846,-86 539.846,-150\"/>\n",
       "<text text-anchor=\"middle\" x=\"451.106\" y=\"-134.8\" font-family=\"Times,serif\" font-size=\"14.00\">worst smoothness &lt;= 0.1786</text>\n",
       "<text text-anchor=\"middle\" x=\"451.106\" y=\"-120.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 12</text>\n",
       "<text text-anchor=\"middle\" x=\"451.106\" y=\"-106.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [3, 9]</text>\n",
       "<text text-anchor=\"middle\" x=\"451.106\" y=\"-92.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 7&#45;&gt;8 -->\n",
       "<g id=\"edge8\" class=\"edge\"><title>7&#45;&gt;8</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M461.36,-185.992C460.102,-177.77 458.723,-168.763 457.396,-160.095\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"460.846,-159.498 455.873,-150.142 453.927,-160.557 460.846,-159.498\"/>\n",
       "</g>\n",
       "<!-- 11 -->\n",
       "<g id=\"node12\" class=\"node\"><title>11</title>\n",
       "<polygon fill=\"#e58139\" fill-opacity=\"0.890196\" stroke=\"black\" points=\"726.001,-150 558.212,-150 558.212,-86 726.001,-86 726.001,-150\"/>\n",
       "<text text-anchor=\"middle\" x=\"642.106\" y=\"-134.8\" font-family=\"Times,serif\" font-size=\"14.00\">worst symmetry &lt;= 0.2682</text>\n",
       "<text text-anchor=\"middle\" x=\"642.106\" y=\"-120.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 20</text>\n",
       "<text text-anchor=\"middle\" x=\"642.106\" y=\"-106.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [18, 2]</text>\n",
       "<text text-anchor=\"middle\" x=\"642.106\" y=\"-92.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 7&#45;&gt;11 -->\n",
       "<g id=\"edge11\" class=\"edge\"><title>7&#45;&gt;11</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M521.794,-185.992C539.449,-176.161 559.12,-165.208 577.355,-155.055\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"579.143,-158.065 586.177,-150.142 575.738,-151.949 579.143,-158.065\"/>\n",
       "</g>\n",
       "<!-- 9 -->\n",
       "<g id=\"node10\" class=\"node\"><title>9</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.890196\" stroke=\"black\" points=\"376.32,-50 279.893,-50 279.893,-0 376.32,-0 376.32,-50\"/>\n",
       "<text text-anchor=\"middle\" x=\"328.106\" y=\"-34.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 10</text>\n",
       "<text text-anchor=\"middle\" x=\"328.106\" y=\"-20.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [1, 9]</text>\n",
       "<text text-anchor=\"middle\" x=\"328.106\" y=\"-6.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 8&#45;&gt;9 -->\n",
       "<g id=\"edge9\" class=\"edge\"><title>8&#45;&gt;9</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M409.121,-85.9375C396.105,-76.3076 381.787,-65.7151 368.856,-56.1483\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"370.707,-53.164 360.587,-50.0301 366.544,-58.7914 370.707,-53.164\"/>\n",
       "</g>\n",
       "<!-- 10 -->\n",
       "<g id=\"node11\" class=\"node\"><title>10</title>\n",
       "<polygon fill=\"#e58139\" stroke=\"black\" points=\"508.202,-50 394.011,-50 394.011,-0 508.202,-0 508.202,-50\"/>\n",
       "<text text-anchor=\"middle\" x=\"451.106\" y=\"-34.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 2</text>\n",
       "<text text-anchor=\"middle\" x=\"451.106\" y=\"-20.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [2, 0]</text>\n",
       "<text text-anchor=\"middle\" x=\"451.106\" y=\"-6.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 8&#45;&gt;10 -->\n",
       "<g id=\"edge10\" class=\"edge\"><title>8&#45;&gt;10</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M451.106,-85.9375C451.106,-77.6833 451.106,-68.7219 451.106,-60.3053\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"454.607,-60.03 451.106,-50.0301 447.607,-60.0301 454.607,-60.03\"/>\n",
       "</g>\n",
       "<!-- 12 -->\n",
       "<g id=\"node13\" class=\"node\"><title>12</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.498039\" stroke=\"black\" points=\"622.32,-50 525.893,-50 525.893,-0 622.32,-0 622.32,-50\"/>\n",
       "<text text-anchor=\"middle\" x=\"574.106\" y=\"-34.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 3</text>\n",
       "<text text-anchor=\"middle\" x=\"574.106\" y=\"-20.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [1, 2]</text>\n",
       "<text text-anchor=\"middle\" x=\"574.106\" y=\"-6.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 11&#45;&gt;12 -->\n",
       "<g id=\"edge12\" class=\"edge\"><title>11&#45;&gt;12</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M618.895,-85.9375C612.179,-76.9496 604.836,-67.1231 598.074,-58.0747\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"600.853,-55.9456 592.063,-50.0301 595.245,-60.1357 600.853,-55.9456\"/>\n",
       "</g>\n",
       "<!-- 13 -->\n",
       "<g id=\"node14\" class=\"node\"><title>13</title>\n",
       "<polygon fill=\"#e58139\" stroke=\"black\" points=\"754.202,-50 640.011,-50 640.011,-0 754.202,-0 754.202,-50\"/>\n",
       "<text text-anchor=\"middle\" x=\"697.106\" y=\"-34.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 17</text>\n",
       "<text text-anchor=\"middle\" x=\"697.106\" y=\"-20.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [17, 0]</text>\n",
       "<text text-anchor=\"middle\" x=\"697.106\" y=\"-6.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 11&#45;&gt;13 -->\n",
       "<g id=\"edge13\" class=\"edge\"><title>11&#45;&gt;13</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M660.88,-85.9375C666.202,-77.133 672.01,-67.5239 677.385,-58.6297\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"680.406,-60.3988 682.583,-50.0301 674.415,-56.778 680.406,-60.3988\"/>\n",
       "</g>\n",
       "<!-- 15 -->\n",
       "<g id=\"node16\" class=\"node\"><title>15</title>\n",
       "<polygon fill=\"#399de5\" stroke=\"black\" points=\"711.32,-243 614.893,-243 614.893,-193 711.32,-193 711.32,-243\"/>\n",
       "<text text-anchor=\"middle\" x=\"663.106\" y=\"-227.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 5</text>\n",
       "<text text-anchor=\"middle\" x=\"663.106\" y=\"-213.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [0, 5]</text>\n",
       "<text text-anchor=\"middle\" x=\"663.106\" y=\"-199.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 14&#45;&gt;15 -->\n",
       "<g id=\"edge15\" class=\"edge\"><title>14&#45;&gt;15</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M663.106,-285.992C663.106,-275.646 663.106,-264.057 663.106,-253.465\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"666.607,-253.288 663.106,-243.288 659.607,-253.288 666.607,-253.288\"/>\n",
       "</g>\n",
       "<!-- 16 -->\n",
       "<g id=\"node17\" class=\"node\"><title>16</title>\n",
       "<polygon fill=\"#e58139\" fill-opacity=\"0.976471\" stroke=\"black\" points=\"903.429,-250 736.784,-250 736.784,-186 903.429,-186 903.429,-250\"/>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-234.8\" font-family=\"Times,serif\" font-size=\"14.00\">worst concavity &lt;= 0.1907</text>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-220.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 137</text>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-206.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [134, 3]</text>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-192.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 14&#45;&gt;16 -->\n",
       "<g id=\"edge16\" class=\"edge\"><title>14&#45;&gt;16</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M712.782,-285.992C728.245,-276.34 745.441,-265.606 761.457,-255.609\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"763.585,-258.407 770.215,-250.142 759.879,-252.469 763.585,-258.407\"/>\n",
       "</g>\n",
       "<!-- 17 -->\n",
       "<g id=\"node18\" class=\"node\"><title>17</title>\n",
       "<polygon fill=\"#399de5\" fill-opacity=\"0.333333\" stroke=\"black\" points=\"895.377,-150 744.836,-150 744.836,-86 895.377,-86 895.377,-150\"/>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-134.8\" font-family=\"Times,serif\" font-size=\"14.00\">worst texture &lt;= 30.975</text>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-120.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 5</text>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-106.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [2, 3]</text>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-92.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 16&#45;&gt;17 -->\n",
       "<g id=\"edge17\" class=\"edge\"><title>16&#45;&gt;17</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M820.106,-185.992C820.106,-177.859 820.106,-168.959 820.106,-160.378\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"823.607,-160.142 820.106,-150.142 816.607,-160.142 823.607,-160.142\"/>\n",
       "</g>\n",
       "<!-- 20 -->\n",
       "<g id=\"node21\" class=\"node\"><title>20</title>\n",
       "<polygon fill=\"#e58139\" stroke=\"black\" points=\"1027.2,-143 913.011,-143 913.011,-93 1027.2,-93 1027.2,-143\"/>\n",
       "<text text-anchor=\"middle\" x=\"970.106\" y=\"-127.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 132</text>\n",
       "<text text-anchor=\"middle\" x=\"970.106\" y=\"-113.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [132, 0]</text>\n",
       "<text text-anchor=\"middle\" x=\"970.106\" y=\"-99.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 16&#45;&gt;20 -->\n",
       "<g id=\"edge20\" class=\"edge\"><title>16&#45;&gt;20</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M867.567,-185.992C885.713,-174.137 906.355,-160.651 924.339,-148.901\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"926.474,-151.687 932.931,-143.288 922.645,-145.827 926.474,-151.687\"/>\n",
       "</g>\n",
       "<!-- 18 -->\n",
       "<g id=\"node19\" class=\"node\"><title>18</title>\n",
       "<polygon fill=\"#399de5\" stroke=\"black\" points=\"868.32,-50 771.893,-50 771.893,-0 868.32,-0 868.32,-50\"/>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-34.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 3</text>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-20.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [0, 3]</text>\n",
       "<text text-anchor=\"middle\" x=\"820.106\" y=\"-6.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = benign</text>\n",
       "</g>\n",
       "<!-- 17&#45;&gt;18 -->\n",
       "<g id=\"edge18\" class=\"edge\"><title>17&#45;&gt;18</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M820.106,-85.9375C820.106,-77.6833 820.106,-68.7219 820.106,-60.3053\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"823.607,-60.03 820.106,-50.0301 816.607,-60.0301 823.607,-60.03\"/>\n",
       "</g>\n",
       "<!-- 19 -->\n",
       "<g id=\"node20\" class=\"node\"><title>19</title>\n",
       "<polygon fill=\"#e58139\" stroke=\"black\" points=\"1000.2,-50 886.011,-50 886.011,-0 1000.2,-0 1000.2,-50\"/>\n",
       "<text text-anchor=\"middle\" x=\"943.106\" y=\"-34.8\" font-family=\"Times,serif\" font-size=\"14.00\">samples = 2</text>\n",
       "<text text-anchor=\"middle\" x=\"943.106\" y=\"-20.8\" font-family=\"Times,serif\" font-size=\"14.00\">value = [2, 0]</text>\n",
       "<text text-anchor=\"middle\" x=\"943.106\" y=\"-6.8\" font-family=\"Times,serif\" font-size=\"14.00\">class = malignant</text>\n",
       "</g>\n",
       "<!-- 17&#45;&gt;19 -->\n",
       "<g id=\"edge19\" class=\"edge\"><title>17&#45;&gt;19</title>\n",
       "<path fill=\"none\" stroke=\"black\" d=\"M862.092,-85.9375C875.108,-76.3076 889.426,-65.7151 902.357,-56.1483\"/>\n",
       "<polygon fill=\"black\" stroke=\"black\" points=\"904.669,-58.7914 910.626,-50.0301 900.506,-53.164 904.669,-58.7914\"/>\n",
       "</g>\n",
       "</g>\n",
       "</svg>\n"
      ],
      "text/plain": [
       "<graphviz.files.Source at 0x11d2ea9b0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import graphviz\n",
    "from IPython.display import display\n",
    "\n",
    "with open(\"tree.dot\") as f:\n",
    "    dot_graph = f.read()\n",
    "\n",
    "display(graphviz.Source(dot_graph))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Feature Importance in trees"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Instead of looking at the whole tree, there are some useful properties that we can derive to summarize the workings of the tree. The most commonly used summary is feature importance, which rates how important each feature is for the decision a tree makes. It is a number between 0 and 1 for each feature, where 0 means “not used at all” and 1 means “perfectly predicts the target.” The feature importances always sum to 1:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "uuid": "dc2f68ee-0df0-47ed-b500-7ec99d5a0a5d"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature importances:\n",
      "[ 0.          0.          0.          0.          0.          0.          0.\n",
      "  0.          0.          0.          0.01019737  0.04839825  0.          0.\n",
      "  0.0024156   0.          0.          0.          0.          0.\n",
      "  0.72682851  0.0458159   0.          0.          0.0141577   0.          0.018188\n",
      "  0.1221132   0.01188548  0.        ]\n"
     ]
    }
   ],
   "source": [
    "print(\"Feature importances:\\n{}\".format(tree.feature_importances_))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To evaluate our supervised models, so far we have split our dataset into a training set and a test set using the train_test_split function, built a model on the training set by calling the fit method, and evaluated it on the test set using the score method, which for classification computes the fraction of correctly classified samples. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Confusion Matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "scikit-learn has its own function for producing confusion matrix. But, let's use pandas which is a popular Python package for data analysis. Its crosstab function produces a better-looking confusion matrix. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set score: 0.988\n",
      "Test set score: 0.951\n",
      "Confusion matrix:\n",
      "Predicted   0   1  All\n",
      "True                  \n",
      "0          49   4   53\n",
      "1           3  87   90\n",
      "All        52  91  143\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "cancer = load_breast_cancer()\n",
    "train_feature, test_feature, train_class, test_class = train_test_split(\n",
    "    cancer.data, cancer.target, stratify=cancer.target, random_state=42)\n",
    "\n",
    "tree = DecisionTreeClassifier(max_depth=4, random_state=0)\n",
    "tree.fit(train_feature, train_class)\n",
    "print(\"Training set score: {:.3f}\".format(tree.score(train_feature, train_class)))\n",
    "print(\"Test set score: {:.3f}\".format(tree.score(test_feature, test_class)))\n",
    "\n",
    "prediction = tree.predict(test_feature)\n",
    "print(\"Confusion matrix:\")\n",
    "print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cross-Validation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The reason we split our data into training and test sets is that we are interested in measuring how well our model generalizes to new, previously unseen data. We are not interested in how well our model fit the training set, but rather in how well it can make predictions for data that was not observed during training.\n",
    " \n",
    "Cross-validation is a statistical method of evaluating generalization performance that is more stable and thorough than using a split into a training and a test set. Cross-validation is implemented in scikit-learn using the cross_val_score function from the model_selection module. The parameters of the cross_val_score function are the model we want to evaluate, the training data, and the ground-truth labels. Let’s evaluate DecisionTreeClassifier on the breast cancer dataset. We can control the number of folds used by setting the cv parameter. We also summarize the cross-validation accuracy by computing the mean accuracy of the multiple folds. \n",
    "\n",
    "scikit-learn uses stratified k-fold cross-validation for classification. In stratified cross-validation, we split the data such that the proportions between classes are the same in each fold as they are in the whole dataset. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validation scores: [ 0.92173913  0.88695652  0.9380531   0.92920354  0.90265487]\n",
      "Average cross-validation score: 0.92\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "cancer = load_breast_cancer()\n",
    "tree = DecisionTreeClassifier(max_depth=4, random_state=0)\n",
    "scores = cross_val_score(tree, cancer.data, cancer.target, cv=5)\n",
    "print(\"Cross-validation scores: {}\".format(scores))\n",
    "print(\"Average cross-validation score: {:.2f}\".format(scores.mean()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Programming Assignment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Dataset and Sample Code"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this assignment, your task is to use classification to predict the quality of wines into 5 different quality levels: 4, 5, 6, 7, 8. You make the classification based on physicochemical tests. The dataset is in a CSV file \"wine.csv\" that is provided to you. The class attribute is the last column \"quality\". The background of the task and the attributes in the dataset are explained in this page: https://archive.ics.uci.edu/ml/datasets/Wine+Quality. \n",
    "\n",
    "For loading CSV file and processing the data, we suggest you to use pandas. A sample program is provided to you, as follows. Make sure to read the comments in the code, which will help you understand it. The sample program works on a dataset \"NBAstats.csv\" which is also provided to you."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']\n",
      "         Player Pos  Age   Tm   G  GS    MP   FG  FGA    FG%  ...     FT%  \\\n",
      "0    Quincy Acy  PF   25  SAC  59  29  14.8  2.0  3.6  0.556  ...   0.735   \n",
      "1  Jordan Adams  SG   21  MEM   2   0   7.5  1.0  3.0  0.333  ...   0.600   \n",
      "2  Steven Adams   C   22  OKC  80  80  25.2  3.3  5.3  0.613  ...   0.582   \n",
      "\n",
      "   ORB  DRB  TRB  AST  STL  BLK  TOV   PF  PS/G  \n",
      "0  1.1  2.1  3.2  0.5  0.5  0.4  0.5  1.7   5.2  \n",
      "1  0.0  1.0  1.0  1.5  1.5  0.0  1.0  1.0   3.5  \n",
      "2  2.7  3.9  6.7  0.8  0.5  1.1  1.1  2.8   8.0  \n",
      "\n",
      "[3 rows x 29 columns]\n",
      "   Age   G  GS    MP   FG  FGA    FG%   3P  3PA    3P%  ...     FT%  ORB  DRB  \\\n",
      "0   25  59  29  14.8  2.0  3.6  0.556  0.3  0.8  0.388  ...   0.735  1.1  2.1   \n",
      "1   21   2   0   7.5  1.0  3.0  0.333  0.0  0.5  0.000  ...   0.600  0.0  1.0   \n",
      "2   22  80  80  25.2  3.3  5.3  0.613  0.0  0.0  0.000  ...   0.582  2.7  3.9   \n",
      "\n",
      "   TRB  AST  STL  BLK  TOV   PF  PS/G  \n",
      "0  3.2  0.5  0.5  0.4  0.5  1.7   5.2  \n",
      "1  1.0  1.5  1.5  0.0  1.0  1.0   3.5  \n",
      "2  6.7  0.8  0.5  1.1  1.1  2.8   8.0  \n",
      "\n",
      "[3 rows x 26 columns]\n",
      "['PF', 'SG', 'C']\n",
      "Test set predictions:\n",
      "['PF' 'SF' 'PF' 'C' 'SF' 'C' 'PF' 'PF' 'SG' 'PG' 'SF' 'SF' 'SG' 'PG' 'SG'\n",
      " 'SG' 'SF' 'C' 'C' 'PF' 'PG' 'SG' 'SF' 'PF' 'SF' 'PG' 'SF' 'SG' 'PG' 'PG'\n",
      " 'SF' 'SF' 'SF' 'PF' 'C' 'PG' 'PG' 'PG' 'SF' 'SF' 'SF' 'C' 'SF' 'PG' 'C'\n",
      " 'C' 'SF' 'PG' 'PG' 'PF' 'PF' 'PF' 'C' 'PG' 'PG' 'PF' 'PG' 'SF' 'SG' 'PF'\n",
      " 'SG' 'C' 'PG' 'PF' 'SF' 'C' 'PF' 'PG' 'PG' 'SF' 'SF' 'C' 'PF' 'PF' 'PF'\n",
      " 'C' 'C' 'PF' 'PG' 'PF' 'PG' 'PF' 'SF' 'SG' 'C' 'C' 'PF' 'SF' 'SF' 'PF'\n",
      " 'PG' 'SG' 'C' 'SG' 'SF' 'C' 'PF' 'PF' 'SG' 'PG' 'PG' 'SG' 'C' 'SF' 'C' 'C'\n",
      " 'PF' 'SF' 'PF' 'SG' 'SG' 'SF' 'PG' 'C' 'PF' 'PG' 'PF' 'PF' 'PF']\n",
      "Test set accuracy: 0.33\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "#read from the csv file and return a Pandas DataFrame.\n",
    "nba = pd.read_csv('NBAstats.csv')\n",
    "\n",
    "# print the column names\n",
    "original_headers = list(nba.columns.values)\n",
    "print(original_headers)\n",
    "\n",
    "#print the first three rows.\n",
    "print(nba[0:3])\n",
    "\n",
    "# \"Position (pos)\" is the class attribute we are predicting. \n",
    "class_column = 'Pos'\n",
    "\n",
    "#The dataset contains attributes such as player name and team name. \n",
    "#We know that they are not useful for classification and thus do not \n",
    "#include them as features. \n",
    "feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \\\n",
    "    '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \\\n",
    "    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']\n",
    "\n",
    "#Pandas DataFrame allows you to select columns. \n",
    "#We use column selection to split the data into features and class. \n",
    "nba_feature = nba[feature_columns]\n",
    "nba_class = nba[class_column]\n",
    "\n",
    "print(nba_feature[0:3])\n",
    "print(list(nba_class[0:3]))\n",
    "\n",
    "train_feature, test_feature, train_class, test_class = \\\n",
    "    train_test_split(nba_feature, nba_class, stratify=nba_class, \\\n",
    "    train_size=0.75, test_size=0.25)\n",
    "\n",
    "training_accuracy = []\n",
    "test_accuracy = []\n",
    "\n",
    "knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)\n",
    "knn.fit(train_feature, train_class)\n",
    "prediction = knn.predict(test_feature)\n",
    "print(\"Test set predictions:\\n{}\".format(prediction))\n",
    "print(\"Test set accuracy: {:.2f}\".format(knn.score(test_feature, test_class)))\n",
    "\n",
    "train_class_df = pd.DataFrame(train_class,columns=[class_column])     \n",
    "train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)\n",
    "train_data_df.to_csv('train_data.csv', index=False)\n",
    "\n",
    "temp_df = pd.DataFrame(test_class,columns=[class_column])\n",
    "temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)\n",
    "test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)\n",
    "test_data_df.to_csv('test_data.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the program above, we use 75% of the data for training and the rest for testing. Note that we are not setting random_state to a fixed value, since this is \"production code\". We built a k nearest neighbor classifier. Also note that we used two more parameters: metric for designating the distance function and p is the exponent in the Minkowski distance function. We used p=1 which means we used the Manhattan distance.\n",
    "\n",
    "To make it easier to understand the constructed classification model, we save the training set into a CSV file and the test set together with the predicted labels into another CSV file. You can look into the CSV files using Microsoft Excel to understand what mistakes were made by the model. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "#### Tasks\n",
    "\n",
    "Your tasks are as follows. In your code, make sure randome_state is not set. \n",
    "\n",
    "1) Use one classification method on the dataset \"wine.csv\". You can apply any of the methods explained in this instruction notebook or any other method in scikit-learn. You can even implement your own method. You can tune your model by using any combination of parameter values. Use 75% of the data for training and the rest for testing.\n",
    "\n",
    "2) Print out the accuracy of the model in 1).\n",
    "\n",
    "3) Print out the confusion matrix for the model in 1). Note that we are dealing with a multi-class (5 quality ratings of wines) classification problem. So the confusion matrix should be 5 x 5. (Actually 6 x 6 since we are also printing the numbers of \"All\". Refer to the earlier example.)\n",
    "\n",
    "4) Use the same model with the same parameters you have chosen in 1). However, instead of using 75%/25% train/test split, apply 10-fold stratified cross-validation. \n",
    "\n",
    "5) Print out the accuracy of each fold in 4).\n",
    "\n",
    "6) Print out the average accuracy across all the folds in 4). "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Grading and Tips\n",
    "\n",
    "For this assignment, the most important thing is to carefully read the instruction notebook and play with the code snippets. The concepts and ideas discussed in the instructions have all been discussed in lectures. Referring to the lecture slides/videos can help you if you face challenges in understanding the instructions. Once you understand the instructions and the code snippets, it won't be difficult to finish the tasks. \n",
    "\n",
    "You will mainly be evaluated on whether you can accomplish the given tasks. Furthermore, 30% of the total score will be based on the accuracy of your model. This way, we believe you can achieve 70% of the score without much struggle and you may have fun trying to improve it. \n",
    "\n",
    "If your method relies on a distance measure, you may consider writing your own distance function, based on your understanding of the data. For instance, KNeighborsClassifier allows you to call your own distance function. \n",
    "\n",
    "To figure out what parameters are available in the various classification methods, you can read more about the specifications of the corresponding Python classes: \n",
    "\n",
    "http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier\n",
    "\n",
    "http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB\n",
    "\n",
    "http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC\n",
    "\n",
    "http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html\n",
    "\n",
    "You can even read the following tutorials about these methods. \n",
    "\n",
    "http://scikit-learn.org/stable/modules/neighbors.html\n",
    "\n",
    "http://scikit-learn.org/stable/modules/naive_bayes.html\n",
    "\n",
    "http://scikit-learn.org/stable/modules/svm.html\n",
    "\n",
    "http://scikit-learn.org/stable/modules/tree.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  },
  "latex_metadata": {
   "author": "Chengkai Li",
   "title": "CSE4334 P2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
