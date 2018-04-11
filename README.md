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

To obtain evaluative measures, I looped through the five subsamples. Of the 5 subsamples, a single subsample is retained as the validation data for testing the model, and the evaluative measures I listed above (precision, recall, etc) were obtained. Then, the AUROC was plotted. In the code below, I used these steps to evaluate Bernoulli Naive Bayes. Similar steps were performed to evaluate Gaussian and Multinomial Naive Bayes.

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

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
