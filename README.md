# ML-Class-KNN-Model
This is a practice of ML-KNN model and understanding the details of KNN

## Dataset
This practice use "Breast Cancer Wisconsin (Original) Data Set" 

url: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

## Question
1.classify whether the subject is "benign" or "malignant" (attribute 11). Implement the
  k-NN classifier for the classification task. To begin one experiment, randomly  draw 
  70 % of the instances from each class for training and the rest are for testing. 
  Repeat the drawing and the k-NN classification 10 times and compute the average 
  accuracy. Then, plot the curve of k versus accuracy for k = 3, ..., 15.

2.compute the covariance matrix of the dataset. The matrix  is of size 9 × 9 
  (attribute 2 – 10). Do you see strong correlation between any two attributes?

## Tips
Because the dataset of "Breast Cancer Wisconsin (Original) Data Set" have some ? value
so we need to check the dataset and let it clean

Randomly draw 70% fot training & 30% for Testing

Repeat 10 times and compute the average accuracy

plot the curve of k versus accuracy for k = 3, ..., 15

## Conclusion

url: https://www.ritchieng.com/machine-learning-k-nearest-neighbors-knn/

(chinese)
url: https://ithelp.ithome.com.tw/articles/10197110 

In this case we don't need LogisticRegression,so we just use the k-nn model.
Because we random split data and find average accurcy,so each operate result and plot 
will be different.

If you don't want to use sklearn model to practice k-nn,you can just use the Euclidean distance 
in your computation. And you need to compute each point distance and find a range depend on 
k value. It might take a lots of time,so if you want to know the K-NN algorithm.
you can read the follow article,it help me a lot.
https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
