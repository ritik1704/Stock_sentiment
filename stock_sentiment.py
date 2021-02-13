import pandas as pd
df=pd.read_csv('Combined_News_DJIA.csv')
train = df[df.Date< '20150101']
test = df[df.Date>'20141214']


data= train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

# there is need to change the names of the coloumns
list= [str(list) for list in range (0,25,1)]
data.columns= list

# now converting news to lower case
for index in data.columns:
    data[index]=data[index].str.lower()
    

# now joining each and every columns into a single one
' '.join(str(x) for x in data.iloc[1,:])

# for all the news into one 
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,:]))
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)

traindataset=cv.fit_transform(headlines).toarray()

# model trained
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(traindataset,train['Label'])

#model test on testing set
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = cv.transform(test_transform).toarray()

#prediction
pred=classifier.predict(test_dataset)


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
report=classification_report(test['Label'],pred)
print(report)

matrix=confusion_matrix(test['Label'],pred)
print(matrix)


score=accuracy_score(test['Label'],pred)
print(score)

