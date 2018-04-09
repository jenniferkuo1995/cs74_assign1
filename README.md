# Assignment 1: Naïve Bayes

Jennifer Kuo, April 9, 2018
CS 74

## 1) Evaluating different variants of Naive Bayes

For this assignment, I tested and evaluated three versions of Naive Bayes (Bernoulli, Gaussian, and Multinomial). 

To do this, I first split my data into X and y. X includes all the features being tested. y has the DV, transformed to be a binary boolean value.

```
import numpy as np
X = np.delete(myData, -1, axis = 1) # select all features
y = myData[:, -1] == 1 #create binary DV
```

Then, I tested each model using K-fold cross validation. I specifically used stratified K-folds, which preserves the proportions of each class.

```
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 5, random_state = 42)
```

For each model, I got the following evaluative measures: precision, recall, F1 score, and AUROC curve.  
```
# accuracy measures: Precision, recall
    precision = []
    recall = []
    f_scores = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    j = 0
    for train_index, test_index in skf.split(X, y):
        C = fn()
        preds = C.fit(X[train_index], y[train_index]).predict(X[test_index])
        probas = C.fit(X[train_index], y[train_index]).predict_proba(X[test_index])
        measures = precision_recall_fscore_support(y[test_index], preds)
        
        precision.append(measures[0])
        recall.append(measures[1])
        f_scores.append(measures[2])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test_index], probas[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (j, roc_auc))
        j += 1

        plt.plot([0, 1], [0, 1], linestyle = '--', lw=2, color='r', alpha = .8)
        
    #### Plotting AUROC
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="right")
    plt.show()

    precision = np.ndarray((5, 2), buffer=np.array(precision))
    recall = np.ndarray((5, 2), buffer=np.array(recall))
    f_scores = np.ndarray((5, 2), buffer=np.array(f_scores))
    return precision, recall, f_scores
```

End with an example of getting some data out of the system or using it for a little demo

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
