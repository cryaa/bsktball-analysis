import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import sklearn.linear_model as linear_model
from sklearn.metrics import log_loss

# data_clean = pd.read_csv('rfselection.csv')
test_data = pd.read_csv('NCAA_Tourney_2018.csv', index_col='game_id')
# datadata = data_clean.drop(['result'],1)
# label = data_clean['result']

data_clean = pd.read_csv('cleandata.csv')
datadata = pd.read_csv('fea_1_train.csv')
label = data_clean['result']

# match test
# datadata = datadata.drop(['Numot', 'team1_score', 'team2_score'], 1)
# print(datadata.head())
# feature selection
datadata = preprocessing.normalize(datadata)

test_data = test_data.drop(['season','team1_position','team2_position','strongseed','weakseed',
                            'team1_region','team2_region','team1_coaches_preseason',
                            'team2_coaches_preseason','team1_coaches_before_final',
                            'team2_coaches_before_final', 'team1_name','slot','host',
                            'team2_name', 'team1_id', 'team2_id'], 1)
test_data['team1_ap_final'].loc[~test_data['team1_ap_final'].isnull()] = 1
test_data['team1_ap_final'].fillna(0, inplace = True)
test_data['team1_ap_preseason'].loc[~test_data['team1_ap_preseason'].isnull()] = 1
test_data['team1_ap_preseason'].fillna(0, inplace = True)
test_data['team2_ap_final'].loc[~test_data['team2_ap_final'].isnull()] = 1
test_data['team2_ap_final'].fillna(0, inplace = True)
test_data['team2_ap_preseason'].loc[~test_data['team2_ap_preseason'].isnull()] = 1
test_data['team2_ap_preseason'].fillna(0, inplace = True)
test_data.to_csv('cleandata18.csv')
# test_data = preprocessing.normalize(test_data)

X_train, X_test, y_train, y_test = train_test_split(datadata, label,
                                                    test_size=0.4, random_state=0)

# random forest
rf = ensemble.RandomForestClassifier(n_estimators=60, bootstrap=True,criterion='entropy',
                                     max_depth=None, max_features=10, min_samples_leaf=3,
                                     min_samples_split=3)
rf.fit(X_train, y_train)
predicted_rf = rf.predict_proba(X_test)
predicted_rf_train = rf.predict_proba(X_train)
predicted_rf1 = rf.predict(X_test)
score_rf = accuracy_score(y_test, predicted_rf1)
print(score_rf)
logl_rf = log_loss(y_test, predicted_rf)
logl_rf_train = log_loss(y_train, predicted_rf_train)
print('train', logl_rf_train,
      'test', logl_rf)

# svm
svc = svm.SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
svc.fit(X_train, y_train)
predicted_svm = svc.predict_proba(X_test)
predicted_svm_train = svc.predict_proba(X_train)
predicted_svm1 = svc.predict(X_test)
score_svm = accuracy_score(y_test, predicted_svm1)
print(score_svm)
logl_svm_train = log_loss(y_train, predicted_svm_train)
logl_svm = log_loss(y_test, predicted_svm)
print('train', logl_svm_train,
      'test', logl_svm)

# logistic regress
logreg = linear_model.LogisticRegression(C=50, multi_class='ovr', penalty= 'l1',
                                         solver = 'liblinear')
logreg.fit(X_train, y_train)
predicted_lr = logreg.predict_proba(X_test)
predicted_lr_train = logreg.predict_proba(X_train)
predicted_lr1 = logreg.predict(X_test)
score_lr = accuracy_score(y_test, predicted_lr1)
print(score_lr)
logl_lr = log_loss(y_test, predicted_lr)
logl_lr_tr = log_loss(y_train, predicted_lr_train)
print('train', logl_lr_tr,
      'test', logl_lr)

# # logistic regress
# logreg = linear_model.LogisticRegression(C=0.01, multi_class='ovr', penalty= 'l1')
# logreg.fit(datadata, label)
# predicted_lr = logreg.predict_proba(test_data)
# # predicted_lr_train = logreg.predict_proba(X_train)
# # predicted_lr1 = logreg.predict(X_test)
# # score_lr = accuracy_score(y_test, predicted_lr1)
# # print(score_lr)
# # logl_lr = log_loss(y_test, predicted_lr)
# # logl_lr_tr = log_loss(y_train, predicted_lr_train)
# # print('train', logl_lr_tr,
# #       'test', logl_lr)
# print(predicted_lr)
