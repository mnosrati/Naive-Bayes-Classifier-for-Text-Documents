# Naive Bayes Classifier for Text Documents

<h3>Introduction</h3>
This is based on a Matlab implementation of “Naive Bayes Classifier for Text Classification” by Masoud Nosrati.

<h3>How to run the program?</h3>
Set the current folder in Matlab on the folder that contains the aforementioned files. Then open “NaiveBayesClassifier.m” and run it.

<h3>Data Set</h3>
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, par- titioned (nearly) evenly across 20 different newsgroups. It was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews[1] paper, though he did not ex- plicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering. The data is organized into 20 different newsgroups.

The original data set is available at http://qwone.com/~jason/20Newsgroups/. In this lab, you won’t need to process the original data set. Instead, a processed version of the data set is provided (see 20newsgroups.zip). This processed version represents 18824 documents which have been divided to two subsets: training (11269 documents) and testing (7505 documents). After unzipping the file, you will find six files: map.csv, train label.csv, train data.csv, test label.csv, test data.csv, vocabulary.txt. The vocabulary.txt contains all distinct words and other tokens in the 18824 documents. train data.csv and test data.csv are format- ted "docIdx, wordIdx, count", where docIdx is the document id, wordIdx represents the word id (in correspondence to vocabulary.txt) and count is the frequency of the word in the document. train label.csv and test label.csv are simply a list of label id’s indicating which newsgroup each document belongs to. The map.csv maps from label id’s to label names.

<h3>Reference</h3>
[1] Ken Lang, Newsweeder: Learning to filter netnews, Proceedings of the Twelfth International Conference on Machine Learning, 331-339 (1995).

