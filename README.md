# Machine Learning Algorithms ⚙️
A repository to document the various machine learning algorithms and their applications in a detailed way
Machine Learning is a subset of Artificial Intelligence(AI) that enables computers to understand data and uncover patterns hidden between them

#### The file 'Machine Learning Algorithms.ipynb' contains the syntax and applications 
#### The individual files contain a problem solved by the algorithm on a dataset

# List of Algorithms
## Linear Regression - Regression
Linear Regression is an algorithm that understands the linear relationship between variables to predict the value of another variable.
1. Contains syntax of creating a Linear Regression Model and predicting using Linear Regression to predict prices of the Boston Housing Dataset and calculating the accuracy.
2. The application of the model using it to predict the salary based on years of experience and understanding by plotting the actual and predicted values

## Logistic Regression - Regression
Logistic Regression is a supervised algorithm used for binary classification tasks.
1. Contains syntax of creating a Logistic Regression Model and predicting using Logistic Regression to predict the outcome of the Cancer Dataset
2. The application of the model using it to predict the outcome of whether the client will subscribe to a term deposit of a bank and calculate the accuracy using some new methods such as classification report and confusion matrix

## Decision Tree 1 - Iterative Dichotomiser 3 - Classification
Iterative Dichotomiser 3 (ID3) recursively splits the dataset based on the feature that provides the highest information gain, creating a tree structure for decision-making.
1. Contains syntax of creating a Decision Tree Model (using entropy - ID3) and to predict the outcome of the Iris Dataset 
2. The application of the algorithm (Iterative Dichotomiser 3 - ID3) using it to predict the type of iris based on entered values and visualizing the data and the Decision tree

## Decision Tree 2 - CART (Classification & Regression Tree) - Classification & Regression
CART splits data based on the Gini Impurity (for classification) or Mean Squared Error (for regression), resulting in binary splits at each node.
1. Contains syntax of creating a Decision Tree Model (using gini impurity) and predicting using a Decision Tree to predict the outcome of the Iris Dataset
2. The application of the algorithm (CART - Classification & Regression Tree) using it to predict the type of iris based on entered values and visualizing the data and the Decision Tree

## Gaussian Naive Bayes Algorithm - Classification
Gaussian Naive Bayes (GNB) is a probabilistic classification algorithm based on Bayes' Theorem, which assumes that the features follow a Gaussian (normal) distribution.
1. Contains syntax of creating a basic Gaussian Naive Bayes Model and predicting the outcome of the Iris dataset
2. The application of the Gaussian Naive Bayes Model and understanding how it predicts the type of iris visualizing the data and calculates the accuracy

## KMeans Clustering Algorithm - Clustering
K-Means Clustering Algorithm iteratively minimizes the within-cluster variance by assigning data points to the nearest cluster centroid and updating the centroids.
1. Contains syntax of creating a basic KMeans Clustering Model and predicting the outcome of the penguin dataset
2. The application of the KMeans Clustering Model and understanding how it predicts the type of penguin visualizing the data and calculating the accuracy using silhouette samples

## KNN (K-Nearest Neighbours) Classifier - Classification
KNN (K-Nearest Neighbours) Classifier predicts the class of a data point based on the majority class of its 𝑘-nearest neighbors in the feature space.
1. Contains syntax of creating a basic KNN(K-Nearest Neighbours) Classifier Model and predicting the outcome of the wine dataset
2. The application of the KNN(K-Nearest Neighbours) Classifier Model and understanding how it predicts the type of wine and visualizing the testing and training accuracy

## KNN (K-Nearest Neighbours) Regressor - Classification
KNN Regressor predicts a continuous value based on the average of the target values of the 𝑘-nearest neighbors.
1. Contains syntax of creating a basic KNN (K-Nearest Neighbours) Regressor Model and predicting the outcome on a dataset created using make_regression.
2. The application of the KNN (K-Nearest Neighbours) Regressor Model and understanding how it predicts the price of a car based on the given values

## Support Vector Classifier (SVC) - Classification
SVC aims to find the hyperplane that best separates data points of different classes in a high-dimensional space while maximizing the margin between the classes.
1. Contains syntax of creating a basic Support Vector Classifier (SVC) Model and predicting the outcome of the breast cancer dataset
2. The application of the Support Vector Classifier (SVC) Model and understanding how it predicts the type of iris and visualizing the data

## Support Vector Regressor (SVR) - Regression
The Support Vector Regressor (SVR) is an extension of the Support Vector Machine (SVM) for regression tasks. Instead of predicting discrete class labels, SVR predicts a continuous output. It tries to fit a function within a margin of tolerance (𝜖) around the true data points, while minimizing the model's complexity.
1. Contains syntax of creating a basic Support Vector Regressor (SVR) Model and predicting the outcome of the diabetes dataset
2. The application of the Support Vector Classifier (SVC) Model and understanding how it predicts the productivity of a worker based on the given features

## Random Forest Classifier - Classification
Random Forest Classifier is an ensemble learning algorithm that combines the predictions of multiple decision trees to classify data points. It leverages the power of randomness to improve accuracy and reduce overfitting.
1. Contains syntax of creating a basic Random Forest Classifier Model and predicting the outcome of the iris dataset
2. The application of the Random Forest Classifier Model and understanding how it predicts the type of penguin and visualizing the data

## Gradient Boosting Classifier - Classification
Gradient Boosting Classifier is an ensemble machine learning algorithm used for classification tasks. It builds a series of decision trees sequentially, where each tree corrects the errors of the previous one, optimizing performance using the concept of gradient descent.
1. Contains the syntax of creating a basic Gradient Boosting Classifier Model and predicting the outcome of the breast cancer dataset
2. The application of the Gradient Boosting Classifier Model and understanding how it predicts the outcome of the PIMA India Diabetes dataset 

## Gradient Boosting Regressor - Regression
The Gradient Boosting Regressor is a powerful ensemble machine learning algorithm used for regression tasks. It builds a model in a stage-wise manner by sequentially adding decision trees that correct the errors of previous models, optimizing for a loss function using gradient descent.
1. Contains the syntax of creating a basic Gradient Boosting Regressor Model and predicting the outcome of a dataset created using the make_regression function of sklearn.datasets
2. The application of the Gradient Boosting Regressor Model and understanding how it predicts the price of a car based on the given values

## Ada Boost Classifier - Classification
The AdaBoost Classifier (Adaptive Boosting) is an ensemble machine learning algorithm that combines multiple weak classifiers (typically decision stumps) to form a strong classifier. It emphasizes training instances that were previously misclassified by assigning them higher weights in subsequent iterations.
1. Contains the syntax of creating a basic Ada Boost Classifier Model and predicting the outcome on a dataset created using make_classification function of sklearn.dataset
2. The application of the Ada Boost Classifier Model and understanding how it predicts if a person will get a stroke or not based on the data provided

## Ada Boost Regressor - Regression
The AdaBoost Regressor (Adaptive Boosting for Regression) is an ensemble machine learning algorithm that combines multiple weak regression models to create a strong predictive model.
1. Contains the syntax of creating an Ada Boost Regressor model and predicting the outcome on a dataset created using make_regression function of sklearn.dataset
2. The application of the Ada Boost Regressor Model and understanding how it predicts the productivity of a worker based on the given features

## XGBoost Classifier - Classification
The XGBoost Classifier (Extreme Gradient Boosting) is a highly efficient and scalable ensemble learning algorithm based on gradient boosting. It is designed for classification tasks and focuses on improving performance and execution speed through optimized implementations and additional features.
1. Contains the syntax and application of the XGBoost Classifier Model predicting the outcome of the iris dataset
   
## XGBoost Regressor - Regression
The XGBoost Regressor is an advanced machine-learning algorithm designed for regression tasks. It is based on gradient boosting and builds an ensemble of decision trees sequentially to minimize a specified loss function. 
1. Contains the syntax and the application of the XGBoost Regressor Model demonstrates predicting housing prices based on features like population and income.
