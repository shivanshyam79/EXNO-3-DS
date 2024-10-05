![318898327-9b500fd0-9b55-4397-b1e8-364652aca983](https://github.com/user-attachments/assets/d16f7579-312e-49b3-a159-574a33ddb687)## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
Developed by :shyam R
Reg No : 212223040200
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
![318691387-9a445ed3-f79e-46ed-8493-a0138abde135](https://github.com/user-attachments/assets/2bdae0cf-52d6-431e-8333-0a5a7c064ece)
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
![318692227-c5ae2314-6f2b-4d93-92b3-f44d1b74015a](https://github.com/user-attachments/assets/7974d978-23a3-46ee-88ea-2c19808fd958)
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
![318692322-4ae17d2a-aa22-4340-9faf-8567549250f6](https://github.com/user-attachments/assets/3108f255-6d6d-4c3e-a83d-b6aab8f24844)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
![318692437-2249ccf3-4a16-462b-b745-677312c7fd42](https://github.com/user-attachments/assets/10780c8c-8449-43c7-8727-a3d732f99da4)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
![318692763-d2714505-ceae-48c6-b428-fc421aaa735d](https://github.com/user-attachments/assets/e235f324-8d54-4665-81d1-31db16c478cf)
df2=pd.concat([df2,enc],axis=1)
df2
![318692827-b4b4c5b2-9bc8-4f41-8649-096999696847](https://github.com/user-attachments/assets/6e1188df-3f99-41e2-8234-752d62c64469)
pd.get_dummies(df2,columns=["nom_0"])
![318692921-e56e11b0-9489-41a5-973c-e32fca8f9840](https://github.com/user-attachments/assets/cb12e175-0d97-46b3-a61e-9908a5bd5bea)
pip install --upgrade category_encoders
![318693032-0711d42f-4456-4222-8334-f183bc7c2385](https://github.com/user-attachments/assets/fb294c58-be12-4024-b003-ba6fb8233541)
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
![318693230-3d2f8b4c-0ffc-4754-8c1b-ad637c727c9b](https://github.com/user-attachments/assets/c3b22808-fe45-464b-b000-54e5f136eec4)
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
![318897767-781ddd71-1fc6-499b-9234-b83778405580](https://github.com/user-attachments/assets/0dd4abd1-e23f-4827-8ada-56d15d31daab)
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
![318897871-6f1877a4-9ba9-45d6-8df2-38fdc103a0ef](https://github.com/user-attachments/assets/dc6e75e4-17a0-4454-9d0c-a69e4b45afc1)
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
![318897982-63cbb12a-e9eb-447e-855a-e56c706bbfa9](https://github.com/user-attachments/assets/e3f4f823-f296-480b-aab9-53dfd11eac37)
df.skew()
![318898092-3d04bbce-76dc-4571-8c8d-5aad234c1766](https://github.com/user-attachments/assets/f3868797-b331-4584-a812-dfc65f5cd0c9)
np.log(df["Highly Positive Skew"])
![318898189-7247340c-6488-4b75-9deb-0ad3f10e03fd](https://github.com/user-attachments/assets/48bfa3d5-cc74-4906-a348-248c60505824)
np.reciprocal(df["Moderate Positive Skew"])
![318898261-71ae0399-a828-406a-93a6-0e36cc31e249](https://github.com/user-attachments/assets/fc848620-0dc7-4770-bc10-bec7c30de7af)
np.sqrt(df["Highly Positive Skew"])
[Uploading 318898327-9b500fd0-9b55-4397-b1e8-364652aca983.png…]()
np.square(df["Highly Positive Skew"])
![318898423-d243323b-c97e-4c55-a41f-f76d176e6461](https://github.com/user-attachments/assets/2c67c452-defd-4413-97ee-bd11d9dff65a)
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
![318898509-758eaaba-b780-4fee-8487-d8242a9d6148](https://github.com/user-attachments/assets/02f30f9f-5ab2-4446-a4f2-696feaad95eb)
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
![318898927-4945b8c6-e27d-4526-9032-0c0aeb9ab576](https://github.com/user-attachments/assets/447dad55-7ad3-49fe-a54c-a691776542e5)
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![318899248-52a7553c-c1bd-4489-a0cb-b13a27684c23](https://github.com/user-attachments/assets/5fc2abc0-19a5-4e24-b44e-466cd2280330)
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
![318899545-3688ed78-4920-4cd4-9e33-4420fc790b8d](https://github.com/user-attachments/assets/4f53a6ca-3528-4cd5-a948-95beca342d5a)
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
### RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully![Uploading 318900112-3987a28b-3816-41b2-9a9d-6a1cedf8382e.png…]()
