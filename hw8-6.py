import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import linear_model
from scipy.stats import bartlett
from scipy.stats import levene
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera
from scipy.stats import normaltest

from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')




#Proje 2'nin normal hali

# df = pd.read_csv("final_dataa.csv")
#
# df['zindexvalue'] = df['zindexvalue'].str.replace(',', '')
# df["zindexvalue"]=df["zindexvalue"].astype(np.int64)
#
# df['price_per_sqft'] = df['lastsoldprice']/df['finishedsqft']
#
# freq = df.groupby('neighborhood').size()
# mean = df.groupby('neighborhood').mean()['price_per_sqft']
# cluster = pd.concat([freq, mean], axis=1)
# cluster['neighborhood'] = cluster.index
# cluster.columns = ['freq', 'price_per_sqft','neighborhood']
#
# cluster1 = cluster[cluster.price_per_sqft < 756]
#
# cluster_temp = cluster[cluster.price_per_sqft >= 756]
# cluster2 = cluster_temp[cluster_temp.freq <123]
#
# cluster3 = cluster_temp[cluster_temp.freq >=123]
#
# def get_group(x):
#     if x in cluster1.index:
#         return 'low_price'
#     elif x in cluster2.index:
#         return 'high_price_low_freq'
#     else:
#         return 'high_price_high_freq'
# df['group'] = df.neighborhood.apply(get_group)
#
# n = pd.get_dummies(df.group)
# df = pd.concat([df, n], axis=1)
# m = pd.get_dummies(df.usecode)
# df = pd.concat([df, m], axis=1)
# drops = ['group', 'usecode']
# df.drop(drops, inplace=True, axis=1)
#
# def is_new(row):
#     if row["yearbuilt"] > 2005:
#         return 1
#     else:
#         return 0
#
# df["is_new"] = df.apply(is_new, axis=1)
#
# y = df['lastsoldprice']
# X = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","totalrooms","yearbuilt","price_per_sqft","is_new","high_price_low_freq","Duplex","MultiFamily2To4"]]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
#
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_preds = regressor.predict(X_test)
#
# print()
# print("ILK Model Ortalama Mutlak Hata (MSE)        : {}".format(mean_absolute_error(y_test, y_preds)))
# print("ILK Model Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_preds)))
# print("ILK Model Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_preds)))
# print("ILK Model Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
# print()
# ########################################################################################################################
# # XGBoosting ile regresyon
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=126)
# xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', seed=126)
#
# xg_reg.fit(X_train,y_train)
# y_tahmin = xg_reg.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_tahmin))
#
# print("XGB Model Ortalama Mutlak Hata (MSE)        : {}".format(mean_absolute_error(y_test, y_tahmin)))
# print("XGB Model Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_tahmin)))
# print("XGB Model Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse))
# print("XGB Model Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_tahmin) / y_test)) * 100))
# #reg:linear depriciate olmuş artık reg:squarederror kullanılıyormuş. O yüzden objective yerine reg:squarederror yazdım.
# #bu xgboost modeli karar ağacı ile çalışıyor.
#
# DM_train = xgb.DMatrix(data = X_train, label=y_train)
# DM_test = xgb.DMatrix(data = X_test, label=y_test)
# params = {"booster":"gblinear", "objective":"reg:squarederror"}
# xg_reg = xgb.train(dtrain=DM_train, params=params, num_boost_round=5)
#
# y_tahmin_gblinear = xg_reg.predict(DM_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_tahmin_gblinear))
# print()
# print("XGB Lineer-Model Ortalama Mutlak Hata (MSE)        : {}".format(mean_absolute_error(y_test, y_tahmin)))
# print("XGB Lineer-Model Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_tahmin)))
# print("XGB Lineer-Model Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse))
# print("XGB Lineer-Model Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_tahmin) / y_test)) * 100))
# #Bu xgboost modeli lineer model kullanıyor. Hata değerleri üstteki karar ağaçıdan daha kötü durumda.
# #Lineer method kullanırken aşağıdaki gibi daha simple bir şekilde yapamaz mıydık?
#
# xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', seed=126, booster="gblinear")
#
# xg_reg.fit(X_train,y_train)
# y_tahmin = xg_reg.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_tahmin))
# print()
# print("XGB Lineer-Model 2 Ortalama Mutlak Hata (MSE)        : {}".format(mean_absolute_error(y_test, y_tahmin)))
# print("XGB Lineer-Model 2 Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_tahmin)))
# print("XGB Lineer-Model 2 Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse))
# print("XGB Lineer-Model 2 Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_tahmin) / y_test)) * 100))
#
# ###################################################################################################################
#
# plt.figure(figsize=(8,8))
# plt.title("'gbtree' ve 'gblinear' karşılaştırması", size = 14)
# ax1 = plt.scatter(y_test, y_tahmin)
# ax2 = plt.scatter(y_test, y_tahmin_gblinear, alpha=0.30)
# ax3 = plt.plot(y_test, y_test, color="red")
# plt.legend((ax1, ax2), ('gbtree', 'gblinear'))
# plt.xlabel("Gerçek Değerler")
# plt.ylabel("Tahmin Değerler")
# plt.show()
#
# df_dmatrix = xgb.DMatrix(data=X, label=y)
# params = {"objective":"reg:squarederror", "max_depth":4}
# ev_fiyatlari_cv = xgb.cv(dtrain=df_dmatrix, params=params, nfold=4,
#                     num_boost_round=100, early_stopping_rounds = 5, metrics="rmse", as_pandas=True, seed=123)
# print(ev_fiyatlari_cv.sort_values(by='test-rmse-mean').head(5))
# # cross validationın en iyi parametreleri.
# # ben bu parametreyi tam olarak nerde kullanacağım konusunu anlayamadım.
#
#
#
#
# df_dmatrix = xgb.DMatrix(data=X, label=y)
# l1_params = np.arange(0.01, 0.2, 0.01)
# params = {"objective":"reg:squarederror","max_depth":3}
# rmses_l1 = []
# for alpha in l1_params:
#     params["alpha"] = alpha
#     cv_rmse = xgb.cv(dtrain=df_dmatrix, params=params, nfold=4, num_boost_round=100,
#                              metrics="rmse", as_pandas=True, early_stopping_rounds=10, seed=123)
#     rmses_l1.append(cv_rmse["test-rmse-mean"].tail(1).values[0])
#
# print("En iyi l1 değerleri:")
# en_iyi_degerler = pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=["l1", "rmse"])
# print(en_iyi_degerler.sort_values('rmse').head(3))
# #Hangi l1 değerinib daha iyi olduğunu bulduk
# # reg_alpha=0.14 diyerek kullanabiliriz.
#
# ####################################################################################################################
# df_dmatrix = xgb.DMatrix(data=X, label=y)
#
# en_iyi_parametreler = {'objective':'reg:squarederror',
#                        'colsample_bytree': 0.7,
#                        'gamma': 0.1,
#                        'learning_rate': 0.3,
#                        'max_depth': 3,
#                        'min_child_weight': 3,
#                        'n_estimators': 50,
#                        'silent':1
#                       }
#
# df_for_xgb = xgb.train(params=en_iyi_parametreler, dtrain=df_dmatrix, num_boost_round=10)
#
# ax = xgb.plot_importance(df_for_xgb)
# ax.figure.set_size_inches(20,8)
# plt.show()
# #Önemli featurelar sıralanıyor.
#
# ax1 = xgb.plot_tree(df_for_xgb, num_trees=5)
# ax1.figure.set_size_inches(30,30)
# plt.show()
# Graphviz packageını çalıştıramadım. Naparsam yapayım
# ""failed to execute ['dot', '-Tpng'], make sure the Graphviz executables are on your systems' PATH "" hatası veriyor.

####################################################################################################################
####################################################################################################################
####################################################################################################################

# Proje 3 Fraud credit card
# Classification
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("creditcard_azaltılmış.csv")

X = df.drop('Class', axis=1)
y = df['Class']

X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.20, random_state=112)

df_dmatrix = xgb.DMatrix(data=X, label=y)



params = {"objective":"reg:logistic", "max_depth":3, "nrounds":8 ,"silent":1}

df_cv = xgb.cv(dtrain=df_dmatrix, params=params, nfold=3,
                    num_boost_round=5, metrics="error", as_pandas=True, seed=124)
print('Doğruluk Değeri : {:.3f}'.format((1-df_cv["test-error-mean"]).max()))








