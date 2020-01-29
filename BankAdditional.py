import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("bank-additional-full.csv", delim_whitespace=False, header=0, sep=";")

# drop column duration as suggest
dataset = dataset.drop(columns=["duration"])

# view dataset
print("Shape: ")
print(dataset.shape)
print()
print("Dataset info: ")
print()
print(dataset.info())

# class distribution
print(dataset.groupby("y").size())

# show feature month - last contact month of year and count it
sns.catplot(x="month", kind="count", data=dataset,
            order=["mar", "apr", "may", "jun", "jul", "aug", "sep", "nov", "dec"])
plt.show()

# count subscribed deposit per mont
ax = sns.countplot(x="month", hue="y", data=dataset,
                   order=["mar", "apr", "may", "jun", "jul", "aug", "sep", "nov", "dec"])
plt.show()

# pdays: number of days that passed by after the client was
# last conctacted from previous campaign (numeric; 99 means client was not previously contacted)
# count y
ax = sns.countplot(x="pdays", hue="y", data=dataset)
plt.show()

# prepare data
dataset_enc = dataset.copy()

# remove columns month and day_of_week
dataset_enc = dataset_enc.drop(columns=["month", "day_of_week"])

# use scikit-learn LabelEncoder to encode labels
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

# convert categorical variable
dataset_enc = pd.get_dummies(dataset_enc, columns=["job"], prefix=["job"])
dataset_enc = pd.get_dummies(dataset_enc, columns=["marital"], prefix=["marital"])
dataset_enc = pd.get_dummies(dataset_enc, columns=["education"], prefix=["education"])
dataset_enc = pd.get_dummies(dataset_enc, columns=["default"], prefix=["default"])
dataset_enc = pd.get_dummies(dataset_enc, columns=["housing"], prefix=["housing"])
dataset_enc = pd.get_dummies(dataset_enc, columns=["loan"], prefix=["loan"])

# binary transform of column contact categorical: "cellular", "telephone"
dataset_enc["contact"] = lb.fit_transform(dataset["contact"])
dataset_enc = pd.get_dummies(dataset_enc, columns=["poutcome"], prefix=["poutcome"])

# move y at end of dataset
dataset_enc["y_class"] = dataset["y"]

# remove original y column
dataset_enc = dataset_enc.drop(columns=["y"])

# view dataset.copy again
dataset_enc.info()

# resample
dataset_majority = dataset_enc[dataset_enc.y_class == "no"]
dataset_minority = dataset_enc[dataset_enc.y_class == "yes"]

# downsample majaority class
from sklearn.utils import resample
df_majority_downsampled = resample(dataset_majority, replace=False, n_samples=4640, random_state=123)

# balanced dataset
dataset_downsampled = pd.concat([df_majority_downsampled, dataset_minority])
dataset_downsampled.y_class.value_counts()
print(dataset_downsampled.y_class.value_counts())

# evaluate algorithm
# split-out validation dataset
array = dataset_downsampled.values
X = array[:, 0:46]
Y = array[:, 46]
validation_size = 0.20
seed = 7

from sklearn.model_selection import train_test_split
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# test options and evaluation metric
num_folds = 10
seed = 7
scoring = "accuracy"

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)

prediction = model.predict(X_validation)
print()
for x in range(2, len(X_validation)):
    print(X_validation[x])
    print(prediction[x])
    print()

# finalize model
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()

# prepare the model
model.fit(X_train, Y_train)
prediction = model.predict(X_validation)

from sklearn.metrics import accuracy_score
print("Accuracy score:", accuracy_score(Y_validation, prediction))

from sklearn.metrics import classification_report
print("Classification report")
print(classification_report(Y_validation, prediction))