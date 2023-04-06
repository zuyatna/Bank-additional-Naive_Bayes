import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Preprocessing
le = preprocessing.LabelEncoder()
dataset = pd.read_csv("bank-additional-full.csv", delim_whitespace=False, header=0, sep=";")

label = le.fit_transform(dataset.y)
job_encoded = le.fit_transform(dataset.job)
marital_encoded = le.fit_transform(dataset.marital)
education_encoded = le.fit_transform(dataset.education)
housing_encoded = le.fit_transform(dataset.housing)
loan_encoded = le.fit_transform(dataset.loan)

tempFeature = []
for i in range(0, len(education_encoded)):
    tempFeature.append([job_encoded[i], marital_encoded[i], education_encoded[i], housing_encoded[i], loan_encoded[i]])

# drop column duration as suggest
dataset = dataset.drop(columns=["duration"])

#view dataset
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
ax = plt.subplot()
ax.set_title('Customers last contact month of year')
plt.show()

# count subscribed deposit per month
ax = sns.countplot(x="month", hue="y", data=dataset,
                   order=["mar", "apr", "may", "jun", "jul", "aug", "sep", "nov", "dec"])
plt.gcf().canvas.manager.set_window_title("Subscribed deposit per month")
plt.show()

# pdays: number of days that passed by after the client was
# last contacted from previous campaign (numeric; 99 means client was not previously contacted)
# count y
ax = sns.countplot(x="pdays", hue="y", data=dataset)
ax.set_title('pdays: Number of days that passed by after the client was last contacted from previous campaign')
plt.show()

# prepare data
dataset_enc = dataset.copy()

# remove columns month and day_of_week
dataset_enc = dataset_enc.drop(columns=["month", "day_of_week"])

# use scikit-learn LabelEncoder to encode labels
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
df_majority_downsampled = resample(dataset_majority)

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

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# test options and evaluation metric
num_folds = 10
seed = 7
scoring = "accuracy"

model = GaussianNB()
model.fit(X_train, Y_train)
prediction = model.predict(X_validation)

# finalize model
model = GradientBoostingClassifier()

# prepare the model
model.fit(X_train, Y_train)
prediction = model.predict(X_validation)

jobOnly = []
for i in range(0, len(tempFeature)):
    temp = tempFeature[i][0]
    jobOnly.append([temp])


modelJob = GaussianNB()
XEdu_train, XEdu_test, yEdu_train, yEdu_test = train_test_split(jobOnly, label, test_size=0.3)  # 70% training and 30% test

modelJob.fit(XEdu_train, yEdu_train)

print("Features Accuracy")

yEdu_pred = modelJob.predict(XEdu_test)
print("Accuracy Job:", metrics.accuracy_score(yEdu_test, yEdu_pred))

maritalOnly = []
for i in range(0, len(tempFeature)):
    temp = tempFeature[i][0]
    maritalOnly.append([temp])


modelMarital = GaussianNB()
XEdu_train, XEdu_test, yEdu_train, yEdu_test = train_test_split(maritalOnly, label, test_size=0.3)  # 70% training and 30% test

modelMarital.fit(XEdu_train, yEdu_train)

yEdu_pred = modelMarital.predict(XEdu_test)
print("Accuracy Marital:", metrics.accuracy_score(yEdu_test, yEdu_pred))

educationOnly = []
for i in range(0, len(tempFeature)):
    temp = tempFeature[i][0]
    educationOnly.append([temp])


modelEducation = GaussianNB()
XEdu_train, XEdu_test, yEdu_train, yEdu_test = train_test_split(educationOnly, label, test_size=0.3)  # 70% training and 30% test

modelEducation.fit(XEdu_train, yEdu_train)

yEdu_pred = modelEducation.predict(XEdu_test)
print("Accuracy Education:", metrics.accuracy_score(yEdu_test, yEdu_pred))

housingOnly = []
for i in range(0, len(tempFeature)):
    temp = tempFeature[i][1]
    housingOnly.append([temp])

modelHousing = GaussianNB()
XEdu_train, XEdu_test, yEdu_train, yEdu_test = train_test_split(housingOnly, label, test_size=0.3)  # 70% training and 30% test

modelHousing.fit(XEdu_train, yEdu_train)

yEdu_pred = modelHousing.predict(XEdu_test)
print("Accuracy Housing:", metrics.accuracy_score(yEdu_test, yEdu_pred))

loanOnly = []
for i in range(0, len(tempFeature)):
    temp = tempFeature[i][2]
    loanOnly.append([temp])

modelLoan = GaussianNB()
XEdu_train, XEdu_test, yEdu_train, yEdu_test = train_test_split(loanOnly, label, test_size=0.3)  # 70% training and 30% test

modelLoan.fit(XEdu_train, yEdu_train)

yEdu_pred = modelLoan.predict(XEdu_test)
print("Accuracy Loan:", metrics.accuracy_score(yEdu_test, yEdu_pred))

print()

print("Accuracy score:", accuracy_score(Y_validation, prediction))
print()

print("Classification report")
print(classification_report(Y_validation, prediction))

# confusion matrix
print("Confusion Matrix: ")
print(confusion_matrix(Y_validation, prediction))

conf_max = confusion_matrix(Y_validation, prediction)
ax = plt.subplot()
sns.heatmap(conf_max, annot=True, ax=ax, fmt='d')

# labels, title and ticks
ax.set_xlabel('Prediction labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['no', 'yes'])
ax.yaxis.set_ticklabels(['no', 'yes'])
plt.show()

