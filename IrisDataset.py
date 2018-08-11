from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

iris = load_iris()

print('type(iris)=',type(iris))

print('iris.feature_names =',iris.feature_names)

print('iris.target_names =',iris.target_names)

print('iris.data =',iris.data)

print('iris.target = ',iris.target)

# Types
print('type(iris.data) =',type(iris.data))

print('type(iris.target) = ',type(iris.target))

features = iris.data
label = iris.target

knnClf = KNeighborsClassifier(n_neighbors=1)
treeClf = tree.DecisionTreeClassifier()

knnClf.fit(features,label)
treeClf.fit(features,label)
print('=============================================================')
print('=============================================================')
print('KNeighborsClassifier')
print('=============================================================')
print('knnClf.predict([[3,4,2,1]]) =',knnClf.predict([[3,4,2,1]]))
print('knnClf.predict([[5,3,3,1]]) =',knnClf.predict([[5,3,3,1]]))
print('knnClf.predict([[6,4,5,2]]) =',knnClf.predict([[6,4,5,2]]))
print('iris.target_names =',iris.target_names)
print('=============================================================')
print('=============================================================')
print('Decision Tree CLassifier')
print('=============================================================')
print('treeClf.predict([[3,4,2,1]]) =',treeClf.predict([[3,4,2,1]]))
print('treeClf.predict([[5,3,3,1]]) =',treeClf.predict([[5,3,3,1]]))
print('treeClf.predict([[6,4,5,2]]) =',treeClf.predict([[6,4,5,2]]))

