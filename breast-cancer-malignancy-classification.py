# Import necessary modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import pandas as pd

# Read the file
breast_cancer = pd.read_csv('data.csv')

# General information
breast_cancer.info()
breast_cancer.head()

# Define the X and y
X = breast_cancer.drop('diagnosis', axis=1)
y = breast_cancer['diagnosis']

# Dealing with null values in the dataset
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
imputer.fit(X)

# New Dataset, without null values
Xtrans = imputer.transform(X)

# Split into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(Xtrans, y, test_size=0.15)

# Initialize the linear svc model
rfc = RandomForestClassifier()

# Checking the score of the model
rfc.fit(xtrain, ytrain)
score = rfc.score(xtest, ytest)
print("Score: ", score)
