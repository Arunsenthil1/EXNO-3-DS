## EXNO-3-DS
```
Name : ARUN S
Reg no : 212224230023
```

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
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="540" height="425" alt="image" src="https://github.com/user-attachments/assets/777c5323-bf74-4859-9eef-feefb02a88ad" />


```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm', 'Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="507" height="197" alt="image" src="https://github.com/user-attachments/assets/6e9f97b3-1809-4ae2-a00f-b340405ead06" />


~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
<img width="593" height="407" alt="image" src="https://github.com/user-attachments/assets/1c6b8f67-593f-4f41-919b-4d4fe121693b" />


```
LABEL ENCODER
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="636" height="412" alt="image" src="https://github.com/user-attachments/assets/035e16e5-db43-402a-bb0a-1eab058482a8" />


```
ONE HOT ENCODER
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc], axis=1)
df2
```
<img width="637" height="402" alt="image" src="https://github.com/user-attachments/assets/182e2803-9629-4594-ae5d-c22be8addc98" />

```
pd.get_dummies (df2,columns=["nom_0"])
```
<img width="921" height="367" alt="image" src="https://github.com/user-attachments/assets/f0355294-c3bb-481b-88e1-404bf4db9f3d" />


```
BINARY ENCODER
df=pd.read_csv("/content/data.csv")
df
```
<img width="1075" height="426" alt="image" src="https://github.com/user-attachments/assets/cfd288ec-1ff5-4e7b-9007-0672c3630bb2" />

```
pip install --upgrade category_encoders
```
<img width="1107" height="300" alt="image" src="https://github.com/user-attachments/assets/dbff5a30-d313-4726-b1c5-6c53ced4b646" />

```
from category_encoders import BinaryEncoder
df
```
<img width="612" height="373" alt="image" src="https://github.com/user-attachments/assets/9739bb29-a24b-49f0-9ca1-7df7aba58169" />

```
be=BinaryEncoder()
nd=be.fit_transform(df ['Ord_2'])
dfb=pd.concat([df,nd], axis=1)
dfb1=df.copy()
dfb
```
<img width="947" height="413" alt="image" src="https://github.com/user-attachments/assets/cb415019-c966-4773-8f37-b59c190d76d4" />

```
TARGET ENCODER
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"], y=cc["Target"])
cc=pd.concat([cc,new], axis=1)
cc
```
<img width="748" height="406" alt="image" src="https://github.com/user-attachments/assets/ba3e5624-a027-40ca-bfa4-9c7e79a44cfb" />


```
FEAUTURE TRANSFORMATION
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("/content/Data_to_Transform.csv") 
df
```
<img width="1025" height="487" alt="image" src="https://github.com/user-attachments/assets/6b7739d8-95f6-4519-9c6e-20b097f5e5c3" />

```
df.skew()
```
<img width="437" height="213" alt="image" src="https://github.com/user-attachments/assets/6d19934d-ba7f-4755-85d4-216b8a167d8d" />

```
np.log(df["Highly Positive Skew"])
```
<img width="382" height="462" alt="image" src="https://github.com/user-attachments/assets/cd5379c6-b492-443b-bc95-f7394bc18d81" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="410" height="466" alt="image" src="https://github.com/user-attachments/assets/ae63f608-f922-499a-9b03-59b75ed212d0" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="392" height="462" alt="image" src="https://github.com/user-attachments/assets/db331f0d-0776-493a-9246-d344fa092a11" />

```
np.square(df["Highly Positive Skew"])
```
<img width="341" height="456" alt="image" src="https://github.com/user-attachments/assets/a9b1bb2d-fbda-48e2-b594-c6f7b5ec2c21" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```
<img width="1087" height="492" alt="image" src="https://github.com/user-attachments/assets/da79d4eb-bd90-4a5b-b59d-5dddd99fef04" />

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"]) 
df.skew()
```
<img width="611" height="272" alt="image" src="https://github.com/user-attachments/assets/848940ae-9e76-4699-b120-ecbe9f0079f6" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```
<img width="522" height="297" alt="image" src="https://github.com/user-attachments/assets/fb41ffc7-0961-4cc0-9bb5-5a946c2362eb" />

```
QUANTILE TRANSFORMATION
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```
<img width="1086" height="517" alt="image" src="https://github.com/user-attachments/assets/daeb19c2-df50-4e79-90a4-5b0e2a2ca68b" />

```
import seaborn as sns 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="763" height="457" alt="image" src="https://github.com/user-attachments/assets/a3ede6c2-e82f-4aca-a80d-5a3bf3cf6d0d" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') 
plt.show()
```
<img width="798" height="457" alt="image" src="https://github.com/user-attachments/assets/ee9e774c-c48c-40d7-a302-ef0e597d92c0" />

```
from sklearn.preprocessing import QuantileTransformer
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="835" height="455" alt="image" src="https://github.com/user-attachments/assets/d5184693-e7a4-4825-8bc2-32a6fa4f3e0b" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'], line='45')
plt.show()
```
<img width="826" height="455" alt="image" src="https://github.com/user-attachments/assets/a3106919-b8e2-4d5d-b8f0-02d41cb70d3b" />


```
sm.qqplot(df['Highly Negative Skew_1'], line='45')
plt.show()
```
<img width="780" height="453" alt="image" src="https://github.com/user-attachments/assets/58d03781-2020-4c9b-a8e8-827dbc4c8d8c" />





















 
       
# RESULT:
   successfully performed Feature Encoding and Transformation process
       
