import pandas as pd

#First we import the data, and check the columns it contains
data = pd.read_csv('data.csv')
print(data.columns)

#We wont be needing majority of these columns so lets cut down to the ones we might be interested in, and see what type of data they are
new_data = data.drop(columns=['Poster_Link','Certificate','Overview','Meta_score','Director','Star1','Star2','Star3','Star4','No_of_Votes'])
print(new_data.dtypes)

#For machine learning, we want to have all our data in numerical form. Here we can see that most of the data are object, or in this case Strings. Even gross which is a number but is represented as a string. 
print(new_data.Genre.head())
new_data = new_data.drop(columns=['Genre'])

#Given the way that the genre column is presented, it probably will be too complicated for now to turn it into usable numbers, for this project we will try to simplify it and use only the remaining ones to determine similarity, Namely: Released Year, Runtime, IMDB Rating, and Gross. Which we will now convert into numerical form
#####    print(new_data.Released_Year.unique())
new_data = new_data[new_data.Released_Year != 'PG']
###    print(new_data.Released_Year.unique())
new_data['New Released'] = new_data.Released_Year.apply(lambda x: int(x))

#Above we noticed that there was a value in Released Year that wasnt a year, it was 'PG', so we removed that and transformed what remained into integers and put that information into a new column called New Released
#print(new_data.Runtime.head())
new_data['New Runtime'] = new_data['Runtime'].apply(lambda x: int(str(x)[:-3]))
#print(new_data.dtypes)
####     print(new_data['New Runtime'])

#Next for Runtime, it contained the minutes, but with the word min next to it. So first we remove the word and leave it as a string with numbers. Then we convert it to integer, put it in a new column called New Runtime, and voila. Runtime has been corrected.
import re 
new_data['New Gross'] = new_data['Gross'].apply(lambda x: re.sub(',','',str(x)))
new_data = new_data[new_data['New Gross'] != 'nan']
new_data['New Gross'] = new_data['New Gross'].apply(lambda x: int(x))
######      print(new_data['New Gross'].unique())

#So now, we removed all the commas from the gross which wouldnt allow us to turn it to integer. We later found that there were some nan values there that werent allowing the data type transformation to occur. So those nans were removed, and the gross was now able to be turned into integers
# It seems like we are all set, now we just move our modified columns into a separate dataframe for convenience sake
df = new_data[['New Gross','New Runtime','IMDB_Rating','New Released','Series_Title']]
#print(df.head())
features = df.drop(columns=['Series_Title','IMDB_Rating'])
labels = df['IMDB_Rating'].apply(lambda x: 1 if x > 7.9 else 0)
print(labels.unique())

#Now we put all the important features into a features dataframe, now we are going to normalize each of the values from 0 to 1, 1 being the highest in that category, and 0 being the lowest in the category. We also create a labels dataframe which is the IMDB Rating, specifically we are trying to predict if, given a set of features, a movie is likely to be really good or not, and we determine really good here as an IMDB greater than 7.9, as that was the most even value. 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
names = features.columns
d = scaler.fit_transform(features)
scaled_features = pd.DataFrame(d, columns=names)
#print(scaled_features.head())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

print(labels.shape)
print(features.shape)

#train_data, train_label, test_data, test_labels = #train_test_split(features, labels, train_size = 0.8, test_size = 0.2)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(features, labels)
y_pred = classifier.predict(features)
print(metrics.accuracy_score(labels,y_pred))

print(labels.value_counts())

#Ok, now that we were able to see the accuracy of our program with 5 nearest neighbors, lets see if by changing the k, we can get better results through a graph!
from matplotlib import pyplot as plt

def Testing_K(features, labels):
  K_Range = range(2,15)
  accuracy = []
  for k in K_Range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(features, labels)
    Y_pred = knn.predict(features)
    accuracy.append(metrics.accuracy_score(labels,Y_pred))

  plt.plot(K_Range, accuracy)
  plt.xlabel('Choice of K')
  plt.ylabel('Accuracy at a given K')
  plt.title('Analyzing K and Accuracy')
  plt.axvline(x = 3, color = 'r')
  plt.show()


Testing_K(features, labels)

#Here we can see that by trying out multiple different k values, it turns out that any k value after 3 seems to decrease performance. So therefore, given the data we were given, by using k nearest neighbors and a k with value of 3, we can expect a 76% accuracy on determining if a movie is going to be higher or lower than 7.9 based on the features used
#This is of course isnt necessarily the case, it could very much be that the test was simply lucky or the features are arbitrary to the point that some by coincidence correlate with each other.
