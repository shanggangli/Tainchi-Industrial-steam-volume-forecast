import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

data_train=pd.read_table(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\工业蒸汽量预测\zhengqi_train.txt',sep="\t")
Train=pd.DataFrame(data_train)
data_test=pd.read_table(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\工业蒸汽量预测\zhengqi_test.txt',sep='\t')
data_test=pd.DataFrame(data_test)
#print(Train.info(verbose=True,null_counts=True))
#print(Train.info(verbose=True,null_counts=True))
data_train["oringin"]="train"
data_test["oringin"]="test"
data_all=pd.concat([data_train,data_test],axis=0,ignore_index=True,sort=False)
'''for column in data_all.columns[0:-2]:
    g = sns.kdeplot(data_all[column][(data_all["oringin"] == "train")], color="Red", shade = True)
    g = sns.kdeplot(data_all[column][(data_all["oringin"] == "test")], ax =g, color="Blue", shade= True)
    g.set_xlabel(column)
    g.set_ylabel("Frequency")
    g = g.legend(["train","test"])
    plt.show()'''
data_all= data_all.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1)
print(data_all.shape)

for column in data_all.columns[0:-2]:
   data_all[column]= data_all[column].map(lambda x:data_all[column].values.mean() if x>=6 or x<=-6 else x)
'''for column in data_all.columns[0:-2]:
   g1=sns.boxplot(data_all[column])
   g1.set_xlabel(column)
   plt.show()'''

# 回归图和直方图
data_train1=data_all[data_all['oringin']=='train'].drop(['oringin'],axis=1)
'''for col in data_train1.columns:
    ax=sns.jointplot(col,"target",data=data_train1,kind='reg',marker='.',height=8, ratio=6, space=0.1)
    plt.show()'''

# 回归图+密度分布+直方图
'''for col in data_train1.columns:
    plt.figure(figsize=(12,8))
    ax=plt.subplot(1,2,1)
    sns.regplot(col,'target',data=data_train1,marker='.',line_kws={'color':'k'})
    ax=plt.subplot(1,2,2)
    sns.distplot(data_train1[col],fit=stats.norm)
    plt.show()'''

# 相关度分析
#plt.figure(figsize=(20,16))
'''columnName=data_train1.columns.tolist()
mcorr=data_train1[columnName].corr(method='spearman') # 相关系数矩阵，即给出了任意两个变量之间的相关系数
mask = np.zeros_like(mcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax=sns.heatmap(mcorr,mask=mask,cmap=cmap,square=True,fmt='0.2f')
plt.show()'''
# 模型
target=data_train['target']
data_all=data_all.drop(['target','oringin'],axis=1)
X=data_all.loc[0:2887,:]

#降维 降维后训练出的效果不好，
'''PCA_model=PCA(n_components=0.98)
PCA_model.fit(X)
X=PCA_model.transform(X)
print(X)
print('贡献值:',PCA_model.explained_variance_ratio_)'''

X_train,X_test,y_train,y_test=train_test_split(X,target,random_state=0,test_size=0.7)
Xgb=xgb.XGBRegressor(max_depth=3, colsample_btree=0.1, learning_rate=0.1, n_estimators=32, min_child_weight=2)
Xgb.fit(X_train,y_train)
Xgb_pred=Xgb.predict(X_test)
MSE=mean_squared_error(y_test,Xgb_pred)
print(MSE)

'''Test=data_all.loc[2888:,:]
Xgb_predtest=Xgb.predict(Test)
result=pd.DataFrame(Xgb_predtest)'''
#result.to_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\工业蒸汽量预测\1.csv',index=False)
