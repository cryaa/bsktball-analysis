import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
import sklearn.linear_model as linear_model
import matplotlib
import numpy as np
from sklearn.metrics import log_loss
# import matplotlib.pyplot as plt


data = pd.read_csv('NCAA_Tourney_2002-2017.csv')
data_clean = data.drop(['Season', 'host_site', 'season', 'team1_coach_id',
                              'team1_coach_name', 'team1_teamname',
           'team2_coach_id','team2_coach_name','team2_teamname', 'team1_id', 'team2_id'], 1)
df_result = pd.DataFrame(data['game_id'])

# -------------------------------------------------------------------------------------------------------------
# col = range(47, len(data_clean.columns.values))
# data_team1 = data_clean.drop(data_clean.columns[col], axis=1)
# def correlation_matrix(df):
#     from matplotlib import pyplot as plt
#     from matplotlib import cm as cm
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     cmap = cm.get_cmap('seismic', 30)
#     cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
#     ax1.grid(True)
#     plt.title('Feature Correlation')
#     labels=[data_team1.columns.values]
#     ax1.set_xticklabels(labels,fontsize=6)
#     ax1.set_yticklabels(labels,fontsize=6)
#     # Add colorbar, make sure to specify tick locations to match desired ticklabels
#     fig.colorbar(cax, ticks=[0,1])
#     plt.show()
# # correlation_matrix(data_team1)
# --------------------------------------------------------------------------------------------------------------
# team1_ap_final
data_clean['team1_ap_final'].loc[~data_clean['team1_ap_final'].isnull()] = 1
data_clean['team1_ap_final'].fillna(0, inplace = True)

# team1_ap_preseason
data_clean['team1_ap_preseason'].loc[~data_clean['team1_ap_preseason'].isnull()] = 1
data_clean['team1_ap_preseason'].fillna(0, inplace = True)

# team1_coaches_before_final
data_clean['team1_coaches_before_final'].loc[~data_clean['team1_coaches_before_final'].isnull()] = 1
data_clean['team1_coaches_before_final'].fillna(0, inplace = True)

# team1_coaches_preseason
data_clean['team1_coaches_preseason'].loc[~data_clean['team1_coaches_preseason'].isnull()] = 1
data_clean['team1_coaches_preseason'].fillna(0, inplace = True)

# team2_ap_final
data_clean['team2_ap_final'].loc[~data_clean['team2_ap_final'].isnull()] = 1
data_clean['team2_ap_final'].fillna(0, inplace = True)

# team2_ap_preseason
data_clean['team2_ap_preseason'].loc[~data_clean['team2_ap_preseason'].isnull()] = 1
data_clean['team2_ap_preseason'].fillna(0, inplace = True)

# team2_coaches_before_final
data_clean['team2_coaches_before_final'].loc[~data_clean['team2_coaches_before_final'].isnull()] = 1
data_clean['team2_coaches_before_final'].fillna(0, inplace = True)

# team2_coaches_preseason
data_clean['team2_coaches_preseason'].loc[~data_clean['team2_coaches_preseason'].isnull()] = 1
data_clean['team2_coaches_preseason'].fillna(0, inplace = True)

data_clean = data_clean.replace('--', '0')
data_clean.to_csv('cleandata.csv', index=None)
# ---------------------------------------------------------------------------------------------------------
ww_feature_sele = data_clean[['result','game_id','team1_adjoe','team1_adjde','team1_coaches_before_final',
                              'team1_coaches_preseason','team1_oppstlrate','team1_oppfg2pct',
                              'team1_oppfg3pct','team1_pt_career_school_losses',
                              'team1_pt_team_season_wins','team1_seed', 'team2_adjoe',
                              'team2_adjde','team2_coaches_before_final',
                              'team2_coaches_preseason','team2_oppstlrate','team2_oppfg2pct',
                              'team2_oppfg3pct','team2_pt_career_school_losses',
                              'team2_pt_team_season_wins','team2_seed']]
ww_feature_sele.to_csv('fea_1_train.csv')
# ---------------------------------------------------------------------------------------------------------
# data_clean = pd.read_csv('nocoach-data.csv')
# # train and test
# datadata = data_clean.drop(['result','Numot', 'team1_score', 'team2_score'], 1)
# label = data_clean['result']
# X_train, X_test, y_train, y_test = train_test_split(datadata, label,
#                                                     test_size=0.5, random_state=0)
#
# # dfdf = pd.DataFrame()
#
# # ---------------------------------------------------------------------------------------------------------
#
# data_clean = pd.read_csv('cleandata.csv')
# datadata = pd.read_csv('fea_1_train.csv')
# label = data_clean['result']
#
# X_train, X_test, y_train, y_test = train_test_split(datadata, label,
#                                                     test_size=0.4, random_state=0)
# data_test = pd.read_csv('NCAA_Tourney_2018.csv')
# test_result = data_test[['game_id']]
# test_data = data_test[['team1_adjoe','team1_adjde','team1_coaches_before_final',
#                               'team1_coaches_preseason','team1_oppstlrate','team1_oppfg2pct',
#                               'team1_oppfg3pct','team1_pt_career_school_losses',
#                               'team1_pt_team_season_wins','team1_seed', 'team2_adjoe',
#                               'team2_adjde','team2_coaches_before_final',
#                               'team2_coaches_preseason','team2_oppstlrate','team2_oppfg2pct',
#                               'team2_oppfg3pct','team2_pt_career_school_losses',
#                               'team2_pt_team_season_wins','team2_seed']]
# test_data['team1_coaches_before_final'].loc[~test_data['team1_coaches_before_final'].isnull()] = 1
# test_data['team1_coaches_before_final'].fillna(0, inplace = True)
# test_data['team1_coaches_preseason'].loc[~test_data['team1_coaches_preseason'].isnull()] = 1
# test_data['team1_coaches_preseason'].fillna(0, inplace = True)
# test_data['team2_coaches_before_final'].loc[~test_data['team2_coaches_before_final'].isnull()] = 1
# test_data['team2_coaches_before_final'].fillna(0, inplace = True)
# test_data['team2_coaches_preseason'].loc[~test_data['team2_coaches_preseason'].isnull()] = 1
# test_data['team2_coaches_preseason'].fillna(0, inplace = True)
# test_result.to_csv('final_result.csv', index=None)
# test_data.to_csv('testdata.csv', index=None)




# # grid search
# clf = ensemble.RandomForestClassifier()
# param_grid = {'n_estimators':range(1,2000,200), "max_depth": [3, None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [2, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}
# grid_search = GridSearchCV(clf, param_grid=param_grid)
# grid_search.fit(X_train,y_train)
# # print(grid_search.best_params_)

# Cs = [0.001, 0.01, 0.1, 1, 10]
# gammas = [0.001, 0.01, 0.1, 1]
# a = ['rbf']
# param_grid = {'C': Cs, 'gamma': gammas, 'kernel': a}
# grid_search = GridSearchCV(svm.SVC(), param_grid)
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)
# #
# param_grid=[{'penalty':['l1','l2'],
#                    'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
#                     'solver':['liblinear'],
#                     'multi_class':['ovr']},
#                 {'penalty':['l2'],
#                  'C':[0.01,0.05,0.1,0.5,1,5,10,50,100],
#                 'solver':['lbfgs'],
#                 'multi_class':['ovr','multinomial']}]
# grid_search = GridSearchCV(linear_model.LogisticRegression(tol=1e-6), param_grid, cv=10)
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)

#
# # random forests
# rf = ensemble.RandomForestClassifier(n_estimators=60, bootstrap=True,criterion='entropy',
#                                      max_depth=None, max_features=10, min_samples_leaf=3,
#                                      min_samples_split=3)
# rf.fit(X_train, y_train)
# predicted_rf = rf.predict_proba(X_test)
# # dfdf['rf_pro'] = predicted_rf[:,1]
# predicted_rf1 = rf.predict(X_test)
# score_rf = accuracy_score(y_test, predicted_rf1)
# print(score_rf)
# logl_rf = log_loss(y_test, predicted_rf)
# print(logl_rf)
#
# # svm
# ll_svm = []
# svc = svm.SVC(kernel='linear', C=0.001, gamma=0.001, probability=True)
# svc.fit(X_train, y_train)
# predicted_svm = svc.predict_proba(X_test)
#
# predicted_svm_train = svc.predict_proba(X_train)
# predicted_svm1 = svc.predict(X_test)
# score_svm = accuracy_score(y_test, predicted_svm1)
# print(score_svm)
# logl_svm_train = log_loss(y_train, predicted_svm_train)
#
# logl_svm = log_loss(y_test, predicted_svm)
# print('train', logl_svm_train,
#       'test', logl_svm)
#
# # dfdf['svm_pro'] = (predicted_svm[:,1].round(3))
#
# # logistic regress
# logreg = linear_model.LogisticRegression()
# logreg.fit(X_train, y_train)
# predicted_lr = logreg.predict_proba(X_test)
# predicted_lr_train = logreg.predict_proba(X_train)
#
# predicted_lr1 = logreg.predict(X_test)
# score_lr = accuracy_score(y_test, predicted_lr1)
# print(score_lr)
# # dfdf['lr_pro'] = (predicted_lr[:,1].round(3))
# logl_lr = log_loss(y_test, predicted_lr)
# logl_lr_tr = log_loss(y_train, predicted_lr_train)
# print('train', logl_lr_tr,
#       'test', logl_lr)
#
#
# # dfdf['nb'] = pd.read_csv('naive bayes.csv', header=None)
#
# # df_result['prob'] = (dfdf.mean(axis=1).round(3))
# # df_result['prob'] = (predicted_lr[:,1].round(3))
#
# # dfdf.to_csv('log.csv')
# # df_result.to_csv('results.csv', index=None)
#
# # df_result['predict'] = df_result['prob'].round(0)
# # df_result['result'] = label
# # df_result.to_csv('results.csv', index=None)
#
# # print(dfdf.mean(axis=1))
# # print(dfdf.mean(axis=1).round(3).mean(axis=0))
# # model = LogisticRegression()
# # # create the RFE model and select 3 attributes
# # rfe = RFE(model, 3)
# # rfe = rfe.fit(X_train, y_train.values.ravel())
# # summarize the selection of the attributes
# # print(rfe.support_)
# # print(rfe.ranking_)
# #
# # from sklearn.ensemble import ExtraTreesClassifier
# #
# # model = ExtraTreesClassifier()
# # model.fit(X_train, y_train.values.ravel())
# # # display the relative importance of each attribute
# # print(model.feature_importances_)