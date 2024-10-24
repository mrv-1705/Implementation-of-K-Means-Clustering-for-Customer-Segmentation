# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start.

Step 2: Load and explore customer data. 

Step 3: Use the Elbow Method to find the best number of clusters. 

Step 4: Perform clustering on customer data. 

Step 5: Plot clustered data to visualize customer segments. 

Step 6: End.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Kurapati Vishnu Vardhan Reddy
RegisterNumber:  212223040103
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/admin/Downloads/Mall_Customers.csv")

data.head()
data.info()
data.isnull().sum()

from sklearn.cluster import KMeans

WCSS = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:, 3:])
    WCSS.append(kmeans.inertia_)

plt.plot(range(1, 11), WCSS)
plt.xlabel("No of Cluster")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters=5)
km.fit(data.iloc[:, 3:])

from sklearn.cluster import KMeans

km = KMeans(n_clusters=5)
y_pred = km.predict(data.iloc[:, 3:])
data["cluster"] = y_pred

df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")

plt.legend()
plt.title("Customer Segments")

```

## Output:

![image](https://github.com/user-attachments/assets/7b36c78c-9bbe-4b3c-b63c-41d914a52373)

Y Predicted Value:

![image](https://github.com/user-attachments/assets/c697cb34-f031-486a-9ab1-212f13b8134f)

Cluster Representation:

![image](https://github.com/user-attachments/assets/47e59901-1f90-4eb5-9f57-56368f9ac91e)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
