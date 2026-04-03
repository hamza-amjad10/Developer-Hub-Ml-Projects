import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_curve,roc_auc_score



df=pd.read_csv("heart_disease_uci.csv")



df.drop(["id","trestbps","chol"],axis=1,inplace=True)

categorical_columns=['fbs',"restecg","exang","slope","thal"]
numeric_columns=["thalch","oldpeak","ca"]

numeric_impute=SimpleImputer(strategy="mean")
categorical_mod=SimpleImputer(strategy="most_frequent")
df[numeric_columns]=numeric_impute.fit_transform(df[numeric_columns])
df[categorical_columns]=categorical_mod.fit_transform(df[categorical_columns])


df["sex"]=df["sex"].map({"Male":1,"Female":0})

df["fbs"] = df["fbs"].map({True:1, False:0})
df["exang"] = df["exang"].map({True:1, False:0})

cate_one_hot=["dataset","cp","restecg","slope","thal"]
numeric_col=["age","thalch","oldpeak","ca"]


df=pd.get_dummies(df,columns=cate_one_hot).astype(float)

X=df.drop("num",axis=1)
Y=df["num"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()

X_train[numeric_col]=scaler.fit_transform(X_train[numeric_col])
X_test[numeric_col]=scaler.transform(X_test[numeric_col])

sm=SMOTE(random_state=42)

X_train_balanaced,Y_train_balanced=sm.fit_resample(X_train,Y_train)

dt=DecisionTreeClassifier(random_state=42)

dt.fit(X_train_balanaced,Y_train_balanced)
y_pred=dt.predict(X_test)
probality=dt.predict_proba(X_test)

print(f"Confusion matrix is: {confusion_matrix(Y_test,y_pred)}")
print(f"Accuracy Score is: {accuracy_score(Y_test,y_pred)}")
print(f"Classification Report is: {classification_report(Y_test,y_pred)}")
print(f"roc curve is: {roc_curve(Y_test,probality[:,1])}")
print(f"roc_auc_score is: {roc_auc_score(Y_test,probality[:,1])}")

