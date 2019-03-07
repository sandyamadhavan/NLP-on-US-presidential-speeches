# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:47:53 2019

@author: Sandya Madhavan - for IST 664,Homework 1

References:Code sample provided in Blackboard

"""

"""********************************Question 2************************************************:
#Analysis of State of the Union Addresses dataset: Part2
#A) Perform the following three tasks (30%, 10% for each task):

#• list the top 50 words by fre quency (normalized by the length of the document)
#• list the top 50 bigrams by frequencies, and
#• list the top 50 bigrams by their Mutual Information scores (using min frequency 5)
"""

#import required nltk package
import nltk
#import freqDist from nltk package
from nltk import FreqDist
###import plaintextcorpusreader to read the text file for processing
from nltk.corpus import PlaintextCorpusReader
mycorpus = PlaintextCorpusReader('.', '.*\.txt')
newstring = mycorpus.raw("state_union_part2.txt")
len(newstring)
#Tokenize the raw text of the state_union_part1.txt file
emmatokens = nltk.word_tokenize(newstring) 
#Convert the words to lower case
emmawords = [w.lower( ) for w in emmatokens]
# show the number of words and print the first 110 words
print(len(emmawords))
print(emmawords[ :110])

# create a frequency distribution of the words, using the NLTK FreqDist module/class, 
# and show the 50 top frequency words. 
#Since the word frequency items are a pair of (word, frequency), 
#we can use item[0] to get the word and item[1] to get the frequency. 
#Printing the string ‘\t’ inserts a tab into the output, so that the frequency numbers line up.

ndist = FreqDist(emmawords)
nitems = ndist.most_common(50)
for item in nitems:
    print (item[0], '\t', item[1])


##filter application to remove punctuations and stop words
#import re package
import re
# this regular expression pattern matches any word that contains all non-alphabetical
#   lower-case characters [^a-z]+
# the beginning ^ and ending $ require the match to begin and end on a word boundary 
pattern = re.compile('^[^a-z]+$')


#function to perform non alphabetical filtering
def alpha_filter(w):
  # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False
    
alphaemmawords = [w for w in emmawords if not alpha_filter(w)]
print(len(alphaemmawords))
print(alphaemmawords[:100])

##stop words filtering
nltk.download('stopwords')
nltkstopwords = nltk.corpus.stopwords.words('english')
#add more stop words
morestopwords = ['could','would','might','must','need','sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve"]

#add more stopwords to create consolidated stop words
stopwords = nltkstopwords + morestopwords

#perform stop word filtering
stoppedemmawords = [w for w in alphaemmawords if not w in stopwords]


#Calculate frequency distribution with our new filtered word list.

emmadist = FreqDist(stoppedemmawords)

emmadist.items()

#for calculating the 50 words by frequency -normalization based on the length of doc
for i in emmadist:
    emmadist[i]/=len(newstring)

emmaitems = emmadist.most_common(50)
print("the list of 50 words by frequency")
for item in emmaitems:
  print(item)
  

###The resultant is the list of the top 50 words by frequency with normalization based on the length of doc
  
'''******************Top 50 bigrams by frequencies*****************'''
  
 #To start using bigrams, we import the collocation finder module. 
from nltk.collocations import *

#Next, for convenience, we define a variable for the bigram measures.
bigram_measures = nltk.collocations.BigramAssocMeasures()

#We start by making an object called a BigramCollocationFinder. 
#The finder then allows us to call other functions to filter the bigrams that it collected and 
#to give scores to the bigrams. We start by scoring the bigrams by frequency by calling the score_ngrams function with the raw_freq scoring measure.

# Note that you must use the entire list of emmawords before any filtering or the raw bigrams may not be adjacent and will not be correct.  Start with all the words and then run the filters in the bigram finder.

finder = BigramCollocationFinder.from_words(emmawords)
scored = finder.score_ngrams(bigram_measures.raw_freq)

#We can see that these scores are sorted into order by decreasing frequency. The scores are the frequencies of the bigrams, normalized to fractions by the total number of bigrams.

for bscore in scored[:50]:
    print (bscore)

#apply alpha_filter created in the previous steps 
    
finder.apply_word_filter(alpha_filter)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:50]:
    print (bscore)

#apply stop words filtering
finder.apply_word_filter(lambda w: w in stopwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print("the 50 bigrams by frequency")
for bscore in scored[:50]:
    print (bscore)
#the resulatnt is the list of the top 50 bigrams by frequencies  
      

    
''' list the top 50 bigrams by their Mutual Information scores (using min frequency 5)'''
#find pmi for the same data as above
#use the same finder,specify min freq as 5.print top 50 bigrams using pmi

finder.apply_freq_filter(5)
scored = finder.score_ngrams(bigram_measures.pmi)
print("the 50 bigrams by frequency 5 and pmi")
for bscore in scored[:50]:
    print (bscore)    

    


