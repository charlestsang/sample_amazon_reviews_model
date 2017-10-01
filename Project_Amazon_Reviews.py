
#!/usr/bin/env python
__author__= "Charles Zeng"
__email__= "charles.x.tsang@gmail.com"
__status__= "Project Example for ML in NLP"
__python_version__= "Python 3.5"

from time import time
start = time()
import pandas as pd
import numpy as np

##########################################################################################################
######################################### Step 1: Basic Explortary Data Analysis #########################
##########################################################################################################
print ('***********************Step 1: Basic EDA ************************')
def Basic_EDA(reviews_data):
    # check basic data parameters:size, dtpyes, column_names
    print(data_frame.info())
    print(data_frame.describe())
    print(data_frame.dtypes)

##########################################################################################################
######################################### Step 2: Features Extraction Raw Dataset ########################
##########################################################################################################
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from ggplot import *

# Create Target Directory called 'sample/'
import os, errno
try:
    os.makedirs('sample/')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

print ('*****************Step 2: Features Extraction ************************')

def Review_Score(reviews_data):
    return reviews_data['RATINGS'].values

# we remove the words that are very commonly used in English
# so we can focus on the important words instead
def remove_stopWords(review):
    word_tokens = []
    stopWords = [] 
    # created empty list to store the reviews filtered from stopwords                                      
    filtered_words = []                                  
    
    #download stopword list from http://algs4.cs.princeton.edu/35applications/stopwords.txt
    stop_words_file =  open("stopwords.txt","r")       

    for each_stop_word in stop_words_file:
        stopWords.append(str(each_stop_word.strip("\n")))     # adding stopwords to stopwords list

    clean_review = str(review)
    # we can add this '.decode('utf-8',errors='ignore')' after 'str()' 
    # if there is any error if the code, for example running in py2.7
    
    # \b[A-Za-z]\b looks for any single letter bounded by word boundaries on both sides.
    processed_review = re.findall(r"\b[a-zA-Z']+\b", clean_review)  
    for each_word in processed_review:
        word_tokens.append(str(each_word))
    upper_Case_stopwords =  map(str.upper,stopWords)

    for word in word_tokens:
        # check condition if word length > 1 and not present in stopwords list
        if (word.lower() not in stopWords and len(word)>1 and word not in upper_Case_stopwords): 
            filtered_words.append(word)
    return filtered_words

def TF_IDF_LatentSemanticAnalysis(reviews_data):  
    review_docs = []
    text_reviews = reviews_data['REVIEW_TEXT']
    for each_review in text_reviews:
        cleaned_review = re.findall(r"\b[a-z']+\b", str(each_review).lower())
        processed_review = ' '.join(word for word in cleaned_review if len(word)>1)
        review_docs.append(str(processed_review))
    stopset = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words= stopset , use_idf= True)
    X = vectorizer.fit_transform(review_docs)
    print ("Features Extraction: TF-IDF is done.")

    number_of_components = 100
    LSA = TruncatedSVD(n_components= number_of_components, algorithm= "randomized", n_iter= 100)
    Reviews_LSA = LSA.fit_transform(X)

    '''
    ########### View LSA Components Details##################
    Re_LSA = LSA.fit(X)
    #print Re_LSA.components_[0]
    terms = vectorizer.get_feature_names()
    # view top 10 key words in 100 componets
    for i, comp in enumerate(Re_LSA.components_): 
        termsInComp = zip (terms,comp)
        sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
        print ("Concept %d:" % i)
        for term in sortedTerms:
            print (term[0])
        print (" ")
    ##########################################################
    '''

    print ("Features Extraction: Computing LSA(Latent_Semantic_Analysis) Component...")
    # store the components vetors in csv file.
    columns_names_list = []
    for i in range(1,number_of_components+1):
        columns_names_list.append("component_"+str(i))
    df1 = pd.DataFrame(Reviews_LSA, columns = columns_names_list )
    df1.to_csv("sample/LSA_Results.csv")
    return df1

def text_features(reviews_data):
    text_reviews = reviews_data['REVIEW_TEXT']
    Review_Lengths_List = []
    Avg_Word_Lengths_List = []
    Avg_Sentence_Lengths_List = []
    Capital_Words_Ratio_List = []
    Question_Exclamation_Ratio_List = []     # ? question mark; ! exclamation mark

    for each_review in text_reviews:
        word_lengths = []
        word_list = remove_stopWords(each_review)
        capital_words_count = 0
        question_exclamation_count = 0

        Review_Lengths_List.append(len(str(each_review).lower()))
        for character in str(each_review):
            if("?" == character or "!" == character):
                question_exclamation_count+=1

        for each_word in word_list:
            word_lengths.append(len(each_word))
            if(str(each_word).isupper() == True):
                capital_words_count+=1

        sentence_lengths = []
        sentences = sent_tokenize(str(each_review),language='english')
        for each_sentence in sentences:
            sentence_lengths.append(len(each_sentence))

        try:
            Avg_Word_Lengths_List.append(sum(word_lengths)/float(len(word_lengths)))
            Avg_Sentence_Lengths_List.append(sum(sentence_lengths)/float(len(sentence_lengths)))
            Capital_Words_Ratio_List.append(capital_words_count/float(len(word_list)))
            Question_Exclamation_Ratio_List.append(question_exclamation_count/float(len(str(each_review))))
        except ZeroDivisionError:
            Avg_Word_Lengths_List.append(0)
            Avg_Sentence_Lengths_List.append(0)
            Capital_Words_Ratio_List.append(0)
            Question_Exclamation_Ratio_List.append(0)

    return Review_Lengths_List , Avg_Word_Lengths_List , Avg_Sentence_Lengths_List , Capital_Words_Ratio_List, Question_Exclamation_Ratio_List


def Create_Helpfulness_Ratio(reviews_data):
    Help_Ratio = []
    for helpful in reviews_data['HELPFUL']:
        #helpful_votes = map(float,helpful.split(','))
        helpful_votes = helpful.split(',')
        Helpfulness_Ratio = float(helpful_votes[0][1:])/float(helpful_votes[1][:-1])
        Help_Ratio.append(Helpfulness_Ratio)  
    return Help_Ratio


def Create_Helpfulness_Class(reviews_data):
    Binary_Class = []
    for helpful in reviews_data['HELPFUL']:
        #helpful_votes = map(float,helpful.split(','))
        helpful_votes = helpful.split(',')
        Helpfulness_Ratio = float(helpful_votes[0][1:])/float(helpful_votes[1][:-1])
        if(Helpfulness_Ratio >= 0.6):
            Binary_Class.append(1)  #Helpful
        else: 
            Binary_Class.append(0)  #unhelpful 
    return Binary_Class


def Extract_Features(review_dataset):
    Review_Rating = Review_Score(review_dataset)
    Review_Length , Avg_Word_Length, Avg_Sentence_Length, Capital_Words_Ratio, Question_Exlamation_Ratio = text_features(review_dataset)
    Class = Create_Helpfulness_Class(review_dataset)
    Ratio = Create_Helpfulness_Ratio(review_dataset)
    return Review_Rating, Review_Length, Avg_Word_Length, Avg_Sentence_Length, Capital_Words_Ratio, Question_Exlamation_Ratio , Class , Ratio

################################################## import sample data  ################################
data_frame = pd.read_csv('sample.csv') #dtype = {'HELPFUL':'float'}
#print (data_frame.head(1))
Basic_EDA(data_frame)

Feature_1, Feature_2, Feature_3, Feature_4, Feature_5, Feature_6, Class, Ratio = Extract_Features(data_frame)

# Assign column 'REVIEW_TEXT' to 'Reviews' as an independt document
Reviews = data_frame['REVIEW_TEXT']

Feature_1_Series = pd.Series(Feature_1)
Feature_2_Series = pd.Series(Feature_2)
Feature_3_Series = pd.Series(Feature_3)
Feature_4_Series = pd.Series(Feature_4)
Feature_5_Series = pd.Series(Feature_5)
Feature_6_Series = pd.Series(Feature_6)
Class_Series     = pd.Series(Class)
Ratio_Series     = pd.Series(Ratio)

df = pd.DataFrame([Feature_1_Series, Feature_2_Series, Feature_3_Series, Feature_4_Series, Feature_5_Series, Feature_6_Series, Class_Series, Ratio_Series])

df2 = df.unstack().unstack()
df2.rename(columns={0:'RW_SCORE',
                    1:'RW_LENGTH',
                    2:'WORD_LENGTH',
                    3:'SENTENCE_LENGTH',
                    4:'CAPS_RATIO',
                    5:'QUES_EXCLAIM_RATIO', 
                    6:'CLASS', 
                    7:'RATIO'}, inplace=True)

df2[['RW_SCORE','RW_LENGTH','WORD_LENGTH','SENTENCE_LENGTH','CAPS_RATIO','QUES_EXCLAIM_RATIO','CLASS', 'RATIO']] = \
        df2[['RW_SCORE','RW_LENGTH','WORD_LENGTH','SENTENCE_LENGTH','CAPS_RATIO','QUES_EXCLAIM_RATIO','CLASS','RATIO']].convert_objects(convert_numeric=True)

df2.to_csv("sample/Raw_Features.csv")

df3 = pd.concat([Reviews,df2],axis=1)
print ("************************* Create basic classifier features *****************************")
#DATA_FOR_LSA = df3.sort('CLASS',ascending = True) 
DATA_FOR_LSA = df3.sort_values(by ='RATIO',ascending = True)#[0:64000]
DATA_FOR_LSA.reset_index(drop=True,inplace=True)
DATA_FOR_LSA.to_csv('sample/Classifier_Features.csv')

print ("************************** Increase Text_Lsa feature into classifier features **************")
LSA_RESULT = TF_IDF_LatentSemanticAnalysis(DATA_FOR_LSA)
df4 = pd.concat([DATA_FOR_LSA, LSA_RESULT],axis=1)
df4.set_index('RW_SCORE',inplace = True)
df4.to_csv('sample/Text_LSA_Features.csv')
#print (df4.head())


##########################################################################################################
###############################Step 3: Advanced Explortary Data Analysis #################################
##########################################################################################################

print ("***************************Step 3: Advanced Explortary Data Analysis ************************")
# EDA from df2: Raw_Features.csv

#print (df2.describe())
print (df2.info())
print (df2.dtypes)
print ("************ Please Close the Plots to Move on...")

# plot all features in dataset
df2.hist()
plt.savefig("data_plot.png", bbox_inches = 'tight')

# plot matrix using scatter_matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df2, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.savefig("scatter_matrix_plot.png")

# boxplot of review_ratio and review_score
df2.boxplot(column='RATIO', by='RW_SCORE', grid=False)
for i in [1,2,3,4,5]:
    y = df2.RATIO[df2.RW_SCORE==i].dropna()
    # Add some random "jitter" to the x-axis
    x = np.random.normal(i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.2)
plt.savefig("box_plot.png")

plt.show()

# measuring features correlations...


###############################################################################################################
######################## Step 4: Supervised Learning: Building Classification Model ###########################
###############################################################################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import metrics

X = pd.read_csv('sample/Classifier_Features.csv')
#select small samples (200 raws) for example
X = X[0:200]
X.fillna(0,inplace=True)
X.drop(['REVIEW_TEXT'],axis=1,inplace=True)
#print X.head()
Y = X.pop('CLASS')  #store label 'CLASS' in Y
numeric_variables = list(X.dtypes[X.dtypes!="object"].index)
X = X[numeric_variables]

cl0 = DecisionTreeClassifier(max_depth=5)
cl1 = RandomForestClassifier(n_estimators= 100 ,criterion="gini", n_jobs= 2)

# use sklearn cross_validation package to fit model
predicted_0 = cross_validation.cross_val_predict(cl0, X, Y, cv=10)
predicted_1 = cross_validation.cross_val_predict(cl1, X, Y, cv=10)

print (" ")
print ("********************Classification Model Results ************************")
print ("--------------------------------------------------------------")
print ("Decision Tree Accuracy: ", metrics.accuracy_score(Y, predicted_0))
print ("Confusion Matrix For Decision Tree Classifier")
print (metrics.confusion_matrix(Y,predicted_0))
print ("AUC Score : ",metrics.roc_auc_score(Y,predicted_0))
print ("Recall : ",metrics.recall_score(Y,predicted_0))
print ("Average Precision Score : ",metrics.average_precision_score(Y,predicted_0))


fpr0, tpr0, _ = metrics.roc_curve(Y, predicted_0, pos_label=None, sample_weight=None, drop_intermediate=True)
#print (predicted_0)
df_DT = pd.DataFrame(dict(fpr=fpr0, tpr=tpr0))
print (df_DT)
p0 = ggplot(df_DT, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
file_name = "roc_curve_decision_tree.png" 
p0.save(file_name)
print (p0)
auc = metrics.auc(fpr0,tpr0)
print(auc)

print ("--------------------------------------------------------------")
print ("Random Forest Accuracy: ", metrics.accuracy_score(Y, predicted_1))
print ("Confusion Matrix For Random Forest Classifier")
print (metrics.confusion_matrix(Y,predicted_1))
print ("AUC Score : ",metrics.roc_auc_score(Y,predicted_1))
print ("Recall : ",metrics.recall_score(Y,predicted_1))
print ("Average Precision Score : ",metrics.average_precision_score(Y,predicted_1))
print (" ")
print ("--------------------------------------------------------------")
fpr1, tpr1, _ = metrics.roc_curve(Y, predicted_1, pos_label=None, sample_weight=None, drop_intermediate=True)
##print (predicted_1)
df_RF = pd.DataFrame(dict(fpr=fpr1, tpr=tpr1))
#print (df_DT)
p1 = ggplot(df_RF, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
file_name = "roc_curve_random_forest.png" 
p1.save(file_name)
print (p1)
auc = metrics.auc(fpr1,tpr1)
print (auc)

###############################################################################################################
######################## Step 5: Unsupervised Learning: Building Regression Model ###########################
###############################################################################################################

#omitting in project example

#################################################### Project End #################################################
print ('Time used {:.2f} s'.format(time()-start))
