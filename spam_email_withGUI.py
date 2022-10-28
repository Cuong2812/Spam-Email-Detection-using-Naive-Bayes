import numpy as np
import pandas as pd
import csv
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import scikitplot as skplt
import nltk

mail_data = pd.read_csv('mail_data.csv')
nltk.download('punkt')
# Word Cloud
ham_words = ''
spam_words = ''
# Creating a corpus of spam messages
for val in mail_data[mail_data['Category'] == 'spam'].Message:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        spam_words = spam_words + words + ' '
# Creating a corpus of ham messages
for val in mail_data[mail_data['Category'] == 'spam'].Message:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '

spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words)
ham_wordcloud = WordCloud(width=500, height=300).generate(ham_words)

#Spam Word cloud
plt.figure( figsize=(7,7), facecolor='w')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#Creating Ham wordcloud
plt.figure( figsize=(7,7), facecolor='w')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#Preproccessing

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

X = mail_data['Message']
Y = mail_data['Category']

#making a donut
amount_of_spam = mail_data['Category'].value_counts()[0]
amount_of_ham = mail_data['Category'].value_counts()[1]
category_names = ['Spam Mail', 'Ham Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colors = ['#ff7675','#a29bfe']
plt.figure(figsize=(2,2), dpi = 230)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle = 90, autopct = '%1.0f%%', colors = custom_colors, pctdistance = 0.8)
centre_circle = plt.Circle((0,0), radius = 0.6, fc = 'white')
plt.gca().add_artist(centre_circle)
plt.show()

#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

#Tokenizing
count_vector = CountVectorizer(lowercase='True',min_df= 1, stop_words='english')
X_train_features = count_vector.fit_transform(X_train)
X_test_features = count_vector.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#alpha=1 means the model uses Laplace Correction
NB_classifier=MultinomialNB(alpha=1, fit_prior='True')
NB_classifier.fit(X_train_features,Y_train)

#Extracting Probability
prob_train = NB_classifier.predict_proba(X_train_features)
prob_spam_train = (prob_train[:,0])
prob_ham_train = (prob_train[:,1])

prob_test = NB_classifier.predict_proba(X_test_features)
prob_spam_test = (prob_test[:,0])
prob_ham_test = (prob_test[:,1])

#Plot Probability of Spam within the training data index
yaxis_label = 'P(X | Spam)'
xaxis_label = 'Email Index'
plt.figure(figsize=(10,6))
plt.xlabel(xaxis_label, fontsize = 14)
plt.ylabel(yaxis_label, fontsize = 14)
email_index=list(range(0,len(prob_spam_train)))
plt.scatter(email_index,prob_spam_train, color = 'blue')
plt.show()

#Evaluating
predict_train = NB_classifier.predict(X_train_features)
skplt.metrics.plot_confusion_matrix(Y_train, predict_train,figsize=(6,6))
plt.show()

predict_test = NB_classifier.predict(X_test_features)
skplt.metrics.plot_confusion_matrix(Y_test, predict_test,figsize=(6,6))
plt.show()

print(classification_report(Y_train, predict_train))
print(classification_report(Y_test, predict_test))


"""Building a Predictive System"""
from tkinter import *
from tkinter import messagebox

window = Tk()
window.title("Spam Email Detection using Naive Bayes")
window.minsize(width=400, height=500)
window.geometry("600x400")

# create the canvas for info
canvas = Canvas(window)
canvas.config()
canvas.pack(expand=True, fill=BOTH)

label = Frame(window)
label.place(relx=0.1, rely=0.15, relwidth=0.8, relheight=0.4)

# email input
text = Text(label,width=80,height=80)
text.pack(padx=20,pady=20)

def email_check():
  input_mail = [text.get("1.0",END)]

  # convert text to feature vectors
  input_data_features = count_vector.transform(input_mail)

  # making prediction

  prediction = NB_classifier.predict(input_data_features)

  if (prediction[0] == 1):
    messagebox.showinfo("Result","This is a HAM email.")

  else:
    messagebox.showinfo("Result","This is a SPAM email.")
# Buttons
checkbtn = Button(window, text="Check", bg="white", fg="black", command=email_check)
checkbtn.place(relx=0.28, rely=0.82, relwidth=0.18, relheight=0.08)

closebtn = Button(window, text="Close", bg="white", fg="black", command=window.destroy)
closebtn.place(relx=0.53, rely=0.82, relwidth=0.18, relheight=0.08)

window.mainloop()
