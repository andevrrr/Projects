import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# Examine the confusion matrix, accuracy_score, recall and precision

df = pd.read_csv('newcar.csv')

# Convert 'Gender' column to numeric, i.e 'Male'-> 1 and 'Female' to 0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# checkinh for missing value and dropping them if there are some
df.isnull().sum()
df.dropna(inplace=True)

X = df.iloc[:, [1, 2, 3]]
y = df.iloc[:, 4]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# for DecisionTreeClassifier

model =  tree.DecisionTreeClassifier(max_depth=100)
model.fit(X_train,y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# make confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy score 
acc = accuracy_score(y_test, y_pred)

# Recall score
rs = recall_score(y_test, y_pred)

# Precision score
ps = precision_score(y_test, y_pred)

print('Accuracy: %.3f' % acc)
print('Recall: %.3f' % rs)
print('Precision: %.3f' % ps)


sns.heatmap(cm, annot=True)
plt.show()