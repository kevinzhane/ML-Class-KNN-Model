import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv("data1.csv")

data.drop("id",axis=1,inplace=True)

df=data.dropna(axis=0)
df.index=range(0,len(df),1)

dtype_object=df.select_dtypes(include=['object'])


## Clear ? 
df["Bare Nuclei"] = df["Bare Nuclei"][df["Bare Nuclei"]!='?']
df.dropna(inplace=True)
df["Bare Nuclei"] = df["Bare Nuclei"].astype("int64")

# create X (features) and y (response)
x = df.drop(["Class"],axis=1)
y = df["Class"].values



# check how many predictions were generated
accuracy = []
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

total_accuracy = []

# Use a loop through the range 3 to 15
for k in range(3,16):
	scores = []
	accuracy = []
	for i in range(1,11):
		
		# STEP 1: split X and y into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

		# STEP 2: Use sklearn k-nn model
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		scores.append(metrics.accuracy_score(y_test, y_pred))
		accuracy.append(scores)

	# STEP 3: Compute the average accuracy
	accuracy = np.mean(accuracy)
	print("Average Accuracy :",accuracy,"K:",k)
	total_accuracy.append(accuracy)

# K=1 through K=13 
k_range = range(3, 16)

# plot the curve plot
plt.subplots(figsize=(10,8))
plt.plot(k_range,total_accuracy)
plt.title('KNN Model practice')
plt.xlabel('Value of K for KNN')
plt.ylabel('KNN Accuracy')
plt.show()

