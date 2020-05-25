
#this classify.py was implemented in jupyter notebook.Please install the required packages.

#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import KFold

# In[2]:
data=pd.read_csv('xxx')

# In[3]:
data.head()


# In[4]:
tamil=pd.DataFrame()
tamil['word']=data['word']
tamil['labels']=data['language']
type(tamil)

# In[5]:
tamil.tail()

# In[6]:
from sklearn.model_selection import train_test_split

# In[7]:
X_train, X_test, y_train, y_test = train_test_split(tamil['word'], tamil['labels'], test_size=0.15, random_state=42)

# In[8]:
type(X_train)

# In[9]:
y_train

# In[10]:
from sklearn.svm import SVC

# In[11]:
from sklearn.feature_extraction.text import TfidfVectorizer

# In[12]:
vectorizer = TfidfVectorizer(analyzer='char')

# In[13]:
xx=vectorizer.fit_transform(X_train)
yy=vectorizer.transform(X_test)

# In[14]:
xx.toarray()

# In[15]:
clf = SVC(gamma='auto')
clf.fit(xx,y_train)

# In[16]:
predicted=clf.predict(yy)

# In[17]:
from sklearn.metrics import accuracy_score

# In[18]:
accuracy_score(y_test,predicted)

# In[19]:
sample=tamil['word'].sample(10)
#list=["மபரப்ரஹ்மணேி","ஸ்வாமி"]
#list
#sample['word']=pd.DataFrame(list)
#t=sample['word']
#type(t)
sample

# In[20]:
vec_sample=vectorizer.transform(sample)
pravin=clf.predict(vec_sample)
pravin


# In[21]:
s="மதிராஸ்"
type(s)

# In[22]:
f = open('/home/grodd/Desktop/sample') 
line = f.readline()
while line:
   
    #print(line)
    
    line = f.readline()
    
f.close()


# In[23]:
file=open('/home/grodd/Desktop/sample',"r")
pra=[]
list=file.readline()
while list:
 
 # print(list)
  temp=list.split(" ")
  #print(type(temp))
  pra.append(temp)
  list=file.readline().replace('\n', '')
  #print(pra)
file.close()

print(pra)


# In[24]:
datas=[]
pravin=pd.DataFrame()

for list in pra:
    for numbers in list:
        n=numbers.rstrip(",\n")
        datas.append(n)
#print(datas)
pravin['words']=datas
print(pravin)
type(pravin)


# In[25]:
sk=pravin['words']
sk
type(sk)


# In[26]:
vec_sample=vectorizer.transform(sk)
pp=clf.predict(vec_sample)
pp


# In[27]:
pravin['labels']=pp


# In[28]:
pravin


# In[29]:
vicky =pravin.loc[pravin['labels']== 1 , ['words']]
v=[]
v = vicky['words']
vicky.to_csv('/home/grodd/Desktop/output.csv', sep='\t', encoding='utf-8')
print (v)


# In[30]:
from sklearn.model_selection import cross_val_score
yz=vectorizer.transform(tamil['word'])
cc=tamil['word']
zz=tamil['labels']
scores = cross_val_score(clf, yz, tamil['labels'], cv=10)
scores                                             


# In[31]:
su=0.0
for s in scores:
    su+=s
su=su/10.0
su


# In[32]:
accuracy=0
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(tamil['word']):
    #print(train)
    X_train, X_test = cc[train_index], cc[test_index]
    y_train, y_test =zz[train_index], zz[test_index]
    xx=vectorizer.fit_transform(X_train)
    yy=vectorizer.transform(X_test)
    clf.fit(xx,y_train)
    predicted=clf.predict(yy)
    accuracy=accuracy+accuracy_score(y_test,predicted)
print(accuracy/10)




