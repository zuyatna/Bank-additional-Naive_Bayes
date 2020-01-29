import pandas as pd

dataset = pd.read_csv('bank-additional-full2.csv')

job = dataset.iloc[:, [1, 2, 3, 5, 6]].values
marital = dataset.iloc[:, 2].values
education = dataset.iloc[:, 3].values
housing = dataset.iloc[:, 5].values
loan = dataset.iloc[:, 6].values

y = dataset.iloc[:, 20].values

print(job)

from sklearn import preprocessing
# buat labelEncoder
le = preprocessing.LabelEncoder()
marital_encoder = le.fit_transform(marital)
housing_encoder = le.fit_transform(housing)
loan_encoder = le.fit_transform(loan)

y_encoded = le.fit_transform(y)
print(y_encoded)

# # splitting dataset into the training set and test set
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(job_encoder, y_encoded, test_size=0.9, random_state=0)
#
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(x_train, y_train)