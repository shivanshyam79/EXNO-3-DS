## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

## FEATURE ENCODING:
Ordinal Encoding An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

Label Encoding Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

Binary Encoding Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

One Hot Encoding We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
## 1. FUNCTION TRANSFORMATION
• Log Transformation • Reciprocal Transformation • Square Root Transformation • Square Transformation

## 2. POWER TRANSFORMATION
• Boxcox method • Yeojohnson method

## CODING AND OUTPUT:
 # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
```
import pandas as pd

df=pd.read_csv("/content/Encoding Data.csv")

df
```
![image](https://github.com/user-attachments/assets/571f35e5-acdf-433e-9c0d-01ca0f0fe089)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/880966da-a533-42be-bcf0-2aab1296b9b4)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])

df
```
![image](https://github.com/user-attachments/assets/8783be74-57b1-45f8-a64d-22c6457b4357)

```
le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc
```
![image](https://github.com/user-attachments/assets/ce5e3e5a-4e23-415f-9c30-89c2a7d27733)

```
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

from sklearn.preprocessing import OneHotEncoder

## Use sparse_output instead of sparse

ohe=OneHotEncoder(sparse_output=False)

df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

df2=pd.concat([df2,enc],axis=1)

df2
```
![image](https://github.com/user-attachments/assets/ebca803c-a426-4d22-bf76-43044f0d4da3)

```
pd.get_dummies(df2,columns=["nom_0"])

![image](https://github.com/user-attachments/assets/085edf39-b19a-48fb-a5a1-71534ef26b5d)

```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/76449c34-113f-4e84-b035-573cc83a5021)

```
from category_encoders import BinaryEncoder

df=pd.read_csv("/content/data.csv")

df
```
![image](https://github.com/user-attachments/assets/1ba3a41e-817e-410b-8d31-194870134df0)

```
be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb
```
![image](https://github.com/user-attachments/assets/fabe7799-1710-4d93-a7eb-56515ed30fd1)

```
from category_encoders import TargetEncoder

te=TargetEncoder()

CC=df.copy()

new=te.fit_transform(X=CC["City"],y=CC["Target"])

CC=pd.concat([CC,new],axis=1)

CC
```
![image](https://github.com/user-attachments/assets/d9809984-5afe-4360-9a0c-e0968ece18f9)

```
import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("/content/Data_to_Transform.csv")

df
```
![image](https://github.com/user-attachments/assets/ab3fae6d-ee58-43b6-8e18-033d114d25ad)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/c5adb13e-77dc-4135-a8a3-9b2f50680db5)
```

np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/0fe646f6-0d8d-4a9c-9d4e-64cd8ca40b60)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/3927b5d6-0810-402c-9ae6-7aa1a853ede2)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/5296b97a-ae8e-455e-8f01-a5bd2315daca)
```

np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/dd4a3d38-9426-440c-b1b0-a79e8d2c88b9)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])

df
```
![image](https://github.com/user-attachments/assets/854a73c3-740b-46e9-9e01-e7a884ec1994)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/user-attachments/assets/9162f132-e205-4929-86c1-60f2a40601c5)

```
import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/61e24e56-3f36-4865-9a9d-d7dd1447c227)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/2b57c33d-a932-40d4-be79-622b864b52c5)

```
from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/3aaba885-f18d-4108-800e-e46dd210ca51)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df["Highly Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/0f18216b-dc7c-48a5-b0d6-be5471855f9c)
```

sm.qqplot(df["Highly Negative Skew_1"],line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/f5822016-d86c-4777-b2d0-b22c43794ebf)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

plt.show()
```
![image](https://github.com/user-attachments/assets/68805865-502a-451b-8723-22e51cda3605)


## RESULT:
   # INCLUDE YOUR RESULT HERE
   Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
