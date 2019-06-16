from flask import Flask,request, render_template
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
import numpy as np 
import os
import pickle
from nltk.stem.snowball import EnglishStemmer
path=os.path.abspath(os.path.dirname(__file__))
tokenizer=nltk.RegexpTokenizer(r'\w+')
stemmer=EnglishStemmer()
donnes=os.path.join(path,'objets')
with open(donnes,'rb') as fichier:
    mon_pickler=pickle.Unpickler(fichier)
    lda=mon_pickler.load()
    frequency_word=mon_pickler.load()
    features=mon_pickler.load()
utilites=os.path.join(path,'utiles')
with open(utilites,'rb') as fich:
    pikler=pickle.Unpickler(fich)
    most_freq=pikler.load()
    tf=pikler.load()
sw=set()

sw.update(most_freq)
app=Flask(__name__)
@app.route('/')
def ind():
   return render_template('index.html')
@app.route('/tags/')
def tag():
   texte=request.args.get('texte')
   resul=''
   if texte is not None:
      texte=str(texte)
      texte=tokenizer.tokenize(texte.lower())
      texte=[stemmer.stem(words) for words in texte]
      texte=[words for words in texte if words not in list(sw)]
      texte=[" ".join(texte)]
      y=tf.transform(texte)
      y=lda.transform(y)
      index=y[0].argsort()[-3:]
      mots=str()   
      for i in index:
          mt=frequency_word[i,:].argsort()[-3:-1]
          for j in mt:
          
             mots+='<'
             mots+=features[j]
             mots+='>'
      resul=mots
   return render_template ('resultat.html',
                           tags=resul
                           )

if __name__=="__main__":
    app.run()
