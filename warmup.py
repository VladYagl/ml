import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, chi2
from sklearn.pipeline import Pipeline


import pandas
import numpy

verbose = False
label_encoder = LabelEncoder()

input_set = pandas.read_csv("train.csv", index_col = 0)
input_target = input_set.values[:, -1]
input_data = input_set.apply(label_encoder.fit_transform).values[:, :-1]

train_data, test_data, train_target, test_target = train_test_split(input_data, input_target, test_size = 0.5, random_state = 1488)
print(train_data.shape, test_data.shape, input_set.shape)

classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(),
        "Linear SVC": LinearSVC(verbose = verbose),
        "SVC": SVC(kernel='linear', verbose = verbose, max_iter = 3000),
}

feature_selectors = {
        "SelectKBest": SelectKBest(chi2, k = 5),
        "L1": SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_data, train_target)),
        # "RFECV": RFECV(estimator = SVC(kernel="linear"), step = 1, cv = 10, scoring = 'accuracy').fit(train_data, train_target),
        # "RFECV + KFold": RFECV(estimator = SVC(kernel="linear"), step = 1, cv = StratifiedKFold(2), scoring = 'accuracy').fit(tain_data, train_target),
        "Tree-based": SelectFromModel(ExtraTreesClassifier(n_estimators = 50).fit(train_data, train_target)),
}

# for clf_name, clf in classifiers.items():
#     for selector_name, selector in feature_selectors.items():
#         name = clf_name  + " + " + selector_name
#         print(name)
#         clf1 = sklearn.base.clone(clf)
#         selector1 = sklearn.base.clone(selector)
#         clf2 = Pipeline([
#             ('feature_selection', selector1),
#             ('classification', clf1)
#         ])
#         clf2.fit(train_data, train_target)
#         scores = cross_val_score(clf2, test_data, test_target, cv = 10)
#         print("%s Accuracy: %0.2f (+/- %0.2f)\n\n" % (name, scores.mean(), scores.std() * 2))

test = pandas.read_csv("test.csv", index_col = 0)
index = test.index
test = test.apply(label_encoder.fit_transform).values
pandas.DataFrame(data = test).to_csv("temp.csv", header = True, index_label = 'id_new')
pandas.DataFrame(data = input_data).to_csv("temp2.csv", header = True, index_label = 'id_new')

print(test.shape)

ans_clf = Pipeline([
    ('feature_selection', feature_selectors['L1']),
    ('classification', classifiers['Random Forest'])
]).fit(train_data, train_target)
ans = ans_clf.predict(test)
pandas.DataFrame(data = ans, index = index, columns = ['class']).to_csv("answer.csv", header = True, index_label = 'id')
