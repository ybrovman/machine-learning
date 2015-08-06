import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
import pandas as pd
import random

def LogLoss(yCV, pred):
    """
    Calculate the logloss evaluation metric.
    LogLoss = 0.6931472 if pred[i] = [0.5, 0.5] for all i.

    :param yCV: ground truth classification labels
    :param pred: prediction probabilities
    :return: logloss
    """
    # using vectorized implementation
    res = np.dot(1-yCV,np.log(pred[:,0])) + np.dot(yCV,np.log(pred[:,1]))
    return -1./len(yCV) * res

def getLogLoss(yCV, pred, threshold = 1e-7):
    """
    Calculate LogLoss function. Use threshold at 1e-7 in order to avoid
    calculating log(0).

    :param yCV: ground truth class labels
    :param pred: prediction propabilities
    :param threshold: small but not zero
    :return: logloss
    """
    pred_prob = []
    for i, val in enumerate(pred):
        c0 = val[0]
        c1 = val[1]
        if c0 < threshold: c0 = threshold
        if c1 < threshold: c1 = threshold
        pred_prob.append([c0,c1])

    pred_prob = np.array(pred_prob)
    return LogLoss(yCV, pred_prob)

def printMetrics(XCV, yCV, clf):
    """
    Print the following classification metrics:
        accuracy, precision, recall, f1 score, LogLoss, AUC score

    :param yCV: cross validation set class labels
    :type  yCV: np.array
    :param XCV: cross validation set feature vector
    :type  XCV: np.array
    :param clf: trained classifier function, probability scores need to be enabled
    :type  clf: sklearn classifier
    :return: performance metrics
    :rtype: tuple
    """
    # print the classifier name
    name = type(clf).__name__
    print '\n', name
    total = len(yCV)
    print 'Size of CV set: ', total

    # print the accuracy
    accuracy = clf.score(XCV, yCV)
    print 'CV Set Accuracy: {:.4f}'.format(accuracy)

    # print precision, recall, f1 score
    pred_prob = clf.predict_proba(XCV)
    pred = np.array([p.argmax() for p in pred_prob])
    print metrics.classification_report(yCV, pred, target_names=['0', '1'])
    mets = metrics.precision_recall_fscore_support(yCV, pred)
    precision = mets[0]
    recall = mets[1]

    # print the LogLoss
    threshold = 1e-7 # needed to avoid taking log(0)
    logloss = getLogLoss(yCV, pred_prob, threshold)
    print 'LogLoss: {:.7f}'.format(logloss)

    # print the AUC score
    pred_prob_POS = pred_prob[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(yCV, pred_prob_POS, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    print 'AUC score: {:.4f}'.format(auc)

    # print number of predicted positive class
    p = 0.5
    positives = sum([1 if i > p else 0 for i in pred_prob[:,1]])
    print 'Number of positive class predictions: ', positives

    return name, total, accuracy, precision, recall, logloss, auc, positives

def makeROC(X, XCV, y, yCV):
    """
    Make ROC plot for several classifiers. Print result metrics.

    :param X: training features
    :type X: np.array
    :param XCV: cross validation features
    :type XCV: np.array
    :param y: training class labels
    :type y: np.array
    :param yCV: cross validation class labels
    :type yCV: np.array
    :return: list of performance metrics
    :rtype: list
    """
    clfs = [GaussianNB(), linear_model.LogisticRegression(), \
            tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier(), \
             ensemble.GradientBoostingClassifier()]

    results = []
    plt.figure(1)
    for i, classifier in enumerate(clfs):
        clf = classifier
        clf.fit(X, y)
        results.append(printMetrics(XCV, yCV, clf))

        pred_prob_POS = clf.predict_proba(XCV)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(yCV, pred_prob_POS, pos_label=1)
        auc = metrics.auc(fpr,tpr)

        # Plot of a ROC curve for a specific class, class type_a
        color = ['g','r','c','m','k']
        plt.plot(fpr, tpr, label=(type(clf).__name__+" = {0:.2f}").format(auc), \
                    color=color[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curves', fontsize=20)
    plt.legend(loc="lower right")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show(block = False)

    return results

def makePrecisionRecallCurve(X, XCV, y, yCV):
    """
    Make precision-recall plot for several classifiers. Print result metrics.

    :param X: training features
    :type X: np.array
    :param XCV: cross validation features
    :type XCV: np.array
    :param y: training class labels
    :type y: np.array
    :param yCV: cross validation class labels
    :type yCV: np.array
    :return: list of performance metrics
    :rtype: list
    """
    clfs = [GaussianNB(), linear_model.LogisticRegression(), \
            tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier(), \
             ensemble.GradientBoostingClassifier()]

    results = []
    plt.figure(2)
    for i, classifier in enumerate(clfs):
        clf = classifier
        clf.fit(X, y)
        results.append(printMetrics(XCV, yCV, clf))

        pred_prob_POS = clf.predict_proba(XCV)[:,1]
        precision, recall, thresholds = metrics.precision_recall_curve(yCV, pred_prob_POS, pos_label=1)

        # Plot of a precision-recall curve for a specific class, class 1
        color = ['g','r','c','m','k']
        plt.plot(recall, precision, label=type(clf).__name__, color=color[i], linewidth=2)

    plt.plot([0, 1], [.5, .5], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('precision-recall curves', fontsize=20)
    plt.legend(loc="lower right")
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show(block = False)

    return results

def removeNeg(x, y, loc):
    if y.size == 1:
        if x < loc: y = 0.0
    else:
        y[np.where(x < loc)] = 0.0
    return y

def cauchyPDF(x, loc, scale):
    return 1./(scale * np.pi * (1.+((x-loc)/scale)**2))

def cauchy(x, mean, std, min):
    return cauchyPDF(x, mean, std)

def gaussianPDF(x, loc, scale):
    return np.exp( - (x - loc) * (x - loc) / (2 * scale * scale)) / np.sqrt(2 * np.pi * scale * scale)

def gaussian(x, mean, std, min):
    return gaussianPDF(x, mean, std)

def exponPDF(x, loc, scale):
    pdf = np.exp( - (x - loc) / scale ) / scale
    return removeNeg(x, pdf, loc)

def expon(x, mean, std, min):
    loc = min
    scale = std
    return exponPDF(x, loc, scale)

def rayleighPDF(x, loc, scale):
    pdf = (x - loc) * np.exp(- (x - loc) * (x - loc) / (2 * scale * scale)) / (scale * scale)
    return removeNeg(x, pdf, loc)

def rayleigh(x, mean, std, min):
    loc = min
    scale = np.sqrt(2./(4. - np.pi)) * std
    return rayleighPDF(x, loc, scale)

def bernoulliPMF(x, p):
    result = x
    if result.size == 1:
        if result == 0 : result = 1 - p
        else: result = p
    else:
        result[np.where(x == 0)] = 1 - p
        result[np.where(x == 1)] = p
    return result

def bernoulli(x, mean, std, min):
    return bernoulliPMF(x, mean)

def NaiveBayesTrain(x, y):
    """
    Train Naive Bayes classifier for binary classes.

    :param X: training features
    :type X: np.array
    :param y: training class labels
    :type y: np.array
    """
    parameterResults = {}
    length = len(x)
    # find prior probabilities p_y0 and p_y1
    p_y1 = sum(y) * 1./length
    p_y0 = 1 - p_y1
    parameterResults["prior"] = (p_y0, p_y1)

    # find parameters for each class
    indexY0 = np.where(y==0)[0]
    indexY1 = np.where(y==1)[0]

    for col in range(x.shape[1]):
        meanX0 = np.mean(x[indexY0,col])
        meanX1 = np.mean(x[indexY1,col])
        stdX0  = np.std(x[indexY0,col])
        stdX1  = np.std(x[indexY1,col])
        minX0  = np.floor(np.min(x[indexY0,col]))
        minX1  = np.floor(np.min(x[indexY1,col]))
        parameterResults["mean"+str(col)] = (meanX0, meanX1)
        parameterResults["std"+str(col)] = (stdX0, stdX1)
        parameterResults["min"+str(col)] = (minX0, minX1)

    return parameterResults

def NaiveBayesPredict(xCV, trainedNB, probFunc):
    predictions = []
    for row in range(len(xCV)):
        condProb0 = 1.0
        condProb1 = 1.0
        for col, xi in enumerate(xCV[row,:]):
            condProb0 *= probFunc(xi, trainedNB["mean"+str(col)][0], trainedNB["std"+str(col)][0], trainedNB["min"+str(col)][0])
            condProb1 *= probFunc(xi, trainedNB["mean"+str(col)][1], trainedNB["std"+str(col)][1], trainedNB["min"+str(col)][1])
        probY0 = trainedNB["prior"][0] * condProb0
        probY1 = trainedNB["prior"][1] * condProb1
        predClass = np.argmax([probY0, probY1])
        predictions.append(predClass)

    return predictions

# make 1 feature from 2 distributions representing 2 classes
np.random.seed(seed=1340)
size0, size1 = 1000, 1000
# c0 = stats.norm.rvs(loc=13, scale = 2, size=size0)
# c1 = stats.norm.rvs(loc=7, scale = 2, size=size1)
c0 = stats.expon.rvs(loc=5, scale = 5, size=size0)
c1 = stats.expon.rvs(loc=0, scale = 5, size=size1)
# c0 = stats.cauchy.rvs(loc=15, scale = .2, size=size0)
# c1 = stats.cauchy.rvs(loc=5, scale = .2, size=size1)
# c0 = stats.rayleigh.rvs(loc=7, scale = 5, size=size0)
# c1 = stats.rayleigh.rvs(loc=0, scale = 5, size=size1)
# c0 = stats.bernoulli.rvs(p = 0.3, size=size0)
# c1 = stats.bernoulli.rvs(p = 0.8, size=size1)

bins = np.linspace(0,20,41)
binWidth = bins[1] - bins[0]
normed = True
if normed: multiplier = binWidth
else: multiplier = 1
vals0, bins0 = np.histogram(c0, bins=bins, normed=normed)
vals1, bins1 = np.histogram(c1, bins=bins, normed=normed)

# add small threshold instead of 0 so KL Divergence does not equal inf
threshold = 1e-7
vals0 = np.array([a if a != 0.0 else threshold for a in vals0])
vals1 = np.array([a if a != 0.0 else threshold for a in vals1])
print "KL Divergence =", stats.entropy(vals0, vals1)

plt.close("all")
plt.bar(bins0[:-1],vals0*multiplier, width=binWidth, alpha=.5, color='b', label="class0")
plt.bar(bins1[:-1],vals1*multiplier, width=binWidth, alpha=.5, color='r', label="class1")

xPDF = np.linspace(0,20,501)
plt.plot(xPDF, gaussianPDF(xPDF, 10, 4.8)*multiplier, label="gaussian", linewidth=2)
# plt.plot(xPDF, cauchyPDF(xPDF, 15, .5)*multiplier, label="cauchy", linewidth=2)
plt.plot(xPDF, exponPDF(xPDF, 5, 4.8)*multiplier, label="expon", linewidth=2)
# plt.plot(xPDF, rayleighPDF(xPDF, 7, 5)*multiplier, label="rayleigh", linewidth=2)
plt.legend(loc='best')
plt.ylim([0, np.ceil(np.max([np.max(vals0*multiplier), np.max(vals1*multiplier)])*100 + 1)/100])
plt.xlabel("feature value")
plt.ylabel("probability density")
# plt.title("FEATURE DISTRIBUTED AS GAUSSIAN DISTRIBUTION")
plt.title("FEATURE DISTRIBUTED AS EXPONENTIAL DISTRIBUTION")
# plt.title("FEATURE DISTRIBUTED AS RAYLEIGH DISTRIBUTION")
plt.show(block=False)

data = np.append(c0, c1).reshape((size0 + size1,1))
labels = np.append([0 for a in range(len(c0))], [1 for a in range(len(c1))])

x, xCV, y, yCV = cv.train_test_split(data, labels, train_size=.8, random_state=42)

# library call to Naive Bayes classifier
clf = GaussianNB()
clf.fit(x,y)
printMetrics(xCV, yCV, clf)
# makeROC(x, xCV, y, yCV)
# makePrecisionRecallCurve(x, xCV, y, yCV)

# compare to Naive Bayes classifier implemented from scratch, using custom conditional probability function
trainedNB = NaiveBayesTrain(x, y)
# predict   = NaiveBayesPredict(xCV, trainedNB, gaussian)
# predict   = NaiveBayesPredict(xCV, trainedNB, cauchy)
predict   = NaiveBayesPredict(xCV, trainedNB, expon)
# predict   = NaiveBayesPredict(xCV, trainedNB, rayleigh)
# predict   = NaiveBayesPredict(xCV, trainedNB, bernoulli)
print "\n>>>>>>>> Manual Naive Bayes Classifier:"
print trainedNB
print 'CV Set Accuracy: {:.4f}'.format(sum(predict==yCV)*1./len(predict))
print metrics.classification_report(yCV, predict, target_names=['0', '1'])
print 'Number of positive class predictions: ', sum(predict)

###################### OUTPUT ######################
### FOR FEATURE DISTRIBUTED AS GAUSSIAN DISTRIBUTION
### CLASS0 = gaussian(loc=13, scale = 2) and CLASS1 = gaussian(loc=7, scale = 2)
'''
KL Divergence = 7.16343882243

GaussianNB
Size of CV set:  400
CV Set Accuracy: 0.9125
             precision    recall  f1-score   support

          0       0.88      0.95      0.92       199
          1       0.95      0.87      0.91       201

avg / total       0.92      0.91      0.91       400

LogLoss: 0.1755118
AUC score: 0.9832
Number of positive class predictions:  184

>>>>>>>> Manual Naive Bayes Classifier:
{'min0': (7.0, 0.0),
'prior': (0.50062499999999999, 0.49937500000000001),
'mean0': (13.011174749174719, 7.0031214951252343),
'std0': (2.0583563293450093, 1.9596600189350077)}
CV Set Accuracy: 0.9125
             precision    recall  f1-score   support

          0       0.88      0.95      0.92       199
          1       0.95      0.87      0.91       201

avg / total       0.92      0.91      0.91       400

Number of positive class predictions:  184
'''

### FOR FEATURE DISTRIBUTED AS EXPONENTIAL DISTRIBUTION
### CLASS0 = expon(loc=5, scale = 5) and CLASS1 = expon(loc=0, scale = 5)
'''
KL Divergence = 1.09021920548

GaussianNB
Size of CV set:  400
CV Set Accuracy: 0.6675
             precision    recall  f1-score   support

          0       0.71      0.57      0.63       199
          1       0.64      0.77      0.70       201

avg / total       0.67      0.67      0.66       400

LogLoss: 0.6121299
AUC score: 0.7975
Number of positive class predictions:  240

>>>>>>>> Manual Naive Bayes Classifier:
{'min0': (5.0, 0.0),
'prior': (0.50062499999999999, 0.49937500000000001),
'mean0': (10.051376964454994, 4.9982876572370456),
'std0': (4.8195716979155829, 4.8426119920999318)}
CV Set Accuracy: 0.8275
             precision    recall  f1-score   support

          0       0.74      1.00      0.85       199
          1       1.00      0.66      0.79       201

avg / total       0.87      0.83      0.82       400

Number of positive class predictions:  132
'''

### FOR FEATURE DISTRIBUTED AS RAYLEIGH DISTRIBUTION
### FOR CLASS0 = rayleigh(loc=7, scale = 5 and CLASS1 = rayleigh(loc=0, scale = 5)
'''
KL Divergence = 2.44698733336

GaussianNB
Size of CV set:  400
CV Set Accuracy: 0.8500
             precision    recall  f1-score   support

          0       0.84      0.86      0.85       199
          1       0.86      0.84      0.85       201

avg / total       0.85      0.85      0.85       400

LogLoss: 0.3588131
AUC score: 0.9182
Number of positive class predictions:  195

>>>>>>>> Manual Naive Bayes Classifier:
{'min0': (7.0, 0.0),
'prior': (0.50062499999999999, 0.49937500000000001),
'mean0': (13.330937105858826, 6.2908049326207482),
'std0': (3.2300162547903364, 3.2262439263150759)}
CV Set Accuracy: 0.8725
             precision    recall  f1-score   support

          0       0.82      0.95      0.88       199
          1       0.95      0.79      0.86       201

avg / total       0.88      0.87      0.87       400

Number of positive class predictions:  168
'''