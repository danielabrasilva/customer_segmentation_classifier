import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Load training dataset
mailout_train = pd.read_csv('data/Udacity_MAILOUT_052018_TRAIN.csv', sep=';', dtype={'CAMEO_DEUG_2015': str, 'CAMEO_INTL_2015':str})
mailout_train = mailout_train.set_index('LNR')

df = pd.read_csv('clustered_df.csv')

# Customers in cleaned dataset
customers_id = np.loadtxt('customers_id.out', dtype=int)
customers = np.intersect1d(customers_id, mailout_train.index.values)

# Create X and y arrays
X = df.loc[customers,:]
y = mailout_train.loc[customers,'RESPONSE']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Logistic Regression classifier
clf = LogisticRegression(solver='liblinear', C=1, max_iter=200)
clf.fit(X_train_scaled, y_train)

y_true = y_train
y_pred = clf.predict(X_train_scaled)
target_names = ['class 0', 'class 1']
print(classification_report(y_true, y_pred, target_names=target_names))
