# Assignment 1: Naïve Bayes

Jennifer Kuo, April 9, 2018
CS 74

## 1) Evaluating different variants of Naive Bayes

For this assignment, I tested and evaluated three versions of Naive Bayes (Bernoulli, Gaussian, and Multinomial). The process for evaluating the three variants was abstracted out into a function called testModel. 

In this function, I first split my data into X (features tested) and y (the DV, transformed to

To do this, I first split my data into X and y. X includes all the features being tested. y has the DV, transformed to be a binary boolean value.

```
X = np.delete(myData, -1, axis = 1) # select all features
y = myData[:, -1] == 1 #create binary DV
```
For each model, I used K-fold cross validation and obtained the following evaluative measures: precision, recall, F1 score, and AUROC curve.  

For cross-validation, data was partitioned into 5 folds using stratifiedKFold, which preserves the proportions of each class.

```
skf = StratifiedKFold(n_splits = 5, random_state = 42)
```

To obtain evaluative measures, I looped through the five subsamples. In each iteration, a single subsample is retained as the validation data. The evaluative measures I listed above (precision, recall, etc) were obtained for each iteration.

```
precisions = []
recalls = []
f_scores = []
aucs = []

# loop through each fold.
j = 0
for train_index, test_index in skf.split(X, y):
        C = model() 
        
        # fit the classifier to the data, excluding one of the subsets. 
        # then, test this classifier on the excluded subset.
        preds = C.fit(X[train_index], y[train_index]).predict(X[test_index])
        probas = C.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
        
        # obtain different evaluative measures
        measures = precision_recall_fscore_support(y[test_index], preds)
        precisions.append(measures[0])
        recalls.append(measures[1])
        f_scores.append(measures[2])
        aucs.append(roc_auc_score(y[test_index], probas[:, 1]))

        j += 1   
```

I then took the means and standard deviations of each evaluative measure and printed these.

```
## obtain mean values of different evaluative measures, and print them out. 
print(r'Mean AUC = %0.2f +/- %0.2f' % (np.mean(aucs), np.std(aucs)))
print(r'Mean precision = %0.2f +/- %0.2f' % (np.mean(precisions), np.std(precisions)))
print(r'Mean recall = %0.2f +/- %0.2f' % (np.mean(recalls), np.std(recalls)))
print('\n')
```

I ran testModel three times, testing the three variants of NB (BernoulliNB, GaussianNB, and MultinomialNB). The following results were obtained. BernoulliNB gave me the best results, so it is what I used for subsequent steps of the assignment.

```
Bernoulli Naive Bayes:
Mean AUC = 0.95 +/- 0.01
Mean precision = 0.88 +/- 0.02
Mean recall = 0.87 +/- 0.06


Gaussian Naive Bayes:
Mean AUC = 0.95 +/- 0.01
Mean precision = 0.83 +/- 0.13
Mean recall = 0.85 +/- 0.11


Multinomial Naive Bayes:
Mean AUC = 0.83 +/- 0.01
Mean precision = 0.77 +/- 0.04
Mean recall = 0.77 +/- 0.08
```

## 2) Data loading, training, testing
In this section, I have code to do the following: 1) data loading 2) training 3) testing, 4) saving the results.

### Functions
I have two functions train(data) and predict(C, row), which abstract out the respective steps.
In train(filename), I read the training data as a nump ndarray (using the numpy function genfromtxt()).
```
myData = np.genfromtxt( filename, delimiter = ',', dtype=np.float64) #read training data
```
I then split this data into two components, which were X (the features used) and y (the DV). With this data, I trained and returned a classifier using BernoulliNB. 

```
return BernoulliNB().fit(X, y)
```

The function predict(C, row) takes a classifier and a row, and uses the classifier to predict either 0 or 1. 
```
def predict(C, row):
    return(C.predict([row]))
```
### Implementing functions

Using my train() function, I built a classifier.

```
myclf = train('train.csv')
```

Next, I loaded test1.csv as a numpy ndarray.
```
testData = np.genfromtxt( 'test1.csv', delimiter = ',', dtype=np.float64)

```
I initialized an empty array called predictions. I then looped through the test data, got predicted DVs for each row (using the predict function), and put these into predictions.
```
predictions = np.zeros(shape =(testData.shape[0], 1))

for i in range(0, testData.shape[0]):
    predictions[i,0] = predict(myclf, testData[i,:])
```

### Saving results
To output my data, I first appended my predictions to the last column of test1.csv. The resulting ndarray was then output as a csv file using the numpy function savetxt.
```
output = np.hstack((testData, predictions)) #append predicted DVs as the last column
np.savetxt("F002341_test1_result.csv", output, delimiter = ",", fmt='%10.5f') # write results as csv

```
