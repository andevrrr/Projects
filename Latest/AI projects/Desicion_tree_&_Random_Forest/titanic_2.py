import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn import ensemble

df = pd.read_csv('titanic.csv')

# checkinh for missing value and dropping them if there are some
df.isnull().sum()
df.dropna(inplace=True)

df = df[['PClass', 'Age', 'Gender', 'Survived']]

X = df.iloc[:, 0:3]
y = df.loc[:, ['Survived']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

model =  ensemble.RandomForestClassifier(max_depth=4)
model.fit(X_train,y_train)
estimator = model.estimators_[5]
# visualize the tree
dot_data = tree.export_graphviz(
            estimator,
            out_file =  None,
            feature_names = list(X.columns),
            class_names = ['Survived','did not survive'],
            filled = True,
            rounded = True)

graph = graphviz.Source(dot_data)
graph.render(filename = 'titanic2', format = 'png')
graph

# Predicting the Test set results
y_pred = model.predict(X_test)

# make confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

# make test people
new_flowers = [{'PClass':1, 'Age':17, 'Gender':1}, #Rose
               {'PClass':3, 'Age':17, 'Gender':0}] #Jack
new_data = pd.DataFrame(new_flowers)

# predict with new data and create dataframe 
new_y = pd.DataFrame(model.predict(new_data))

# apply species information based on the prediction
new_y[1] = new_y[0].apply(lambda x: 'did not survive' if x == 0  else 'Survived' )