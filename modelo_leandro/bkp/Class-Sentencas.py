#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import nltk
import random
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import pickle
import pyodbc

from sklearn.pipeline import Pipeline


# In[ ]:





# In[7]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[8]:


#dftotal = pd.DataFrame()
#from glob import glob 
#Arquivo= '/mnt/dados/Sentencas/dadossentenca' 
#arquivos = glob(Arquivo)
df = pd.read_csv('dadossentenca/SentencaCorpus.csv', sep="|", encoding='utf-8')


# In[9]:


dfS=df.loc[(df['Label']==True)]
dfN=df.loc[(df['Label']==False)]
AmostraN=dfN.sample(n=80000,random_state=random.randint(1,100), replace=False)


# In[10]:


dfS = dfS.append(AmostraN)


# In[ ]:


dfS


# In[11]:


X =dfS['Sentenca'].str.lower()
Y = dfS['Label']


# In[12]:


X=X.fillna('')
Y=Y.fillna('')


# In[14]:


stopwords = set(stopwords.words('portuguese') + list(punctuation))
X_Limpo = X.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
Y_Limpo = Y  #Y.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))


# In[44]:


X_train, X_test, Y_train, Y_test = train_test_split(X_Limpo, Y_Limpo, test_size=.30, random_state=random.randint(1,100), shuffle=True)


# In[ ]:





# In[45]:


tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=3)


# In[46]:


tf_train = tfidf.fit_transform(X_train)
tf_test = tfidf.transform(X_test)


# In[ ]:


svc = SVC(class_weight='balanced', kernel='linear', C=1)


# In[ ]:


svc.partial_fit(tf_train, Y_train)
pred = svc.predict(tf_test)


# In[ ]:


####### PRINTAR MATRIX #############
print("SVC-->>",classification_report(y_pred=pred, y_true=Y_test))
matrix = confusion_matrix(y_pred=pred, y_true=Y_test)
plot_confusion_matrix(matrix, ['S', 'N'])


# In[ ]:


########### TESTAR ######
pred=svc.predict(tfidf.transform(['Extinta a Execução/Cumprimento da Sentença pela Satisfação da Obrigação    Vistos. Ante a manifestação do credor de fls. 45, julgo extinta esta ação de procedimento comum, ora em fase de Execução de Título Extrajudicial, em que são partes Condomínio Edifício Palmeiras Imperiais e Maria Luiza Machado Pedrosa e outro, o que faço com fundamento na norma do artigo 924, II, do CPC. Considerando a inequívoca manifestação do credor, que expressamente reconheceu a integral satisfação do crédito, ato incompatível com a intenção de recorrer, declaro o imediato trânsito em julgado. Oportunamente, remetam-se os autos ao arquivo, observadas as formalidades legais e cautelas de praxe. Publique-se e intime-se.']))
print(pred)


# In[ ]:


pipeline = Pipeline([
    ('feature', tfidf),
    ('classifier', svc)
])


# In[ ]:


pred = svc.predict(tf_test)


# In[ ]:


model = pipeline.fit(X_train, Y_train)


# In[ ]:


model.score(X_train, Y_train)


# In[ ]:


from joblib import dump

dump(model, 'EvidenciaSentenca.joblib')


# In[17]:


#******************** outro NÂO USADO AINDA ***************
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# In[31]:


tf_train=tf_train.reshape(1, -1)


# In[47]:


n_estimators = 10
#start = time.time()
clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=10))


# In[48]:


clf.fit(tf_train, Y_train)


# In[51]:


clf.score(tf_test, Y_test)


# In[79]:


texto=['Processo nº 0000001-66.2009.8.18.0042 Classe: Execução de Título Extrajudicial Exequente: ROYTON QUIMICA FARMACEUTICA LTDA Advogado(s): DOMINGOS GUSTAVO DE SOUZA(OAB/SÃO PAULO Nº 26283) Executado(a): F P RODRIGUES Advogado(s): SENTENÇA (...) Assim, impõe-se a extinção do processo por restar evidenciada a falta de utilidade e/ou necessidade na sua continuidade, fazendo desaparecer uma das condições da ação, no caso, o interesse processual. Pelo exposto, com fundamento no art. 485, III e VI do NCPC, julgo extinto o processo sem exame do mérito. Condeno a parte autora no pagamento das custas processuais. Sem condenação em honorários. Publique-se. Registre-se. Intimem-se. BOM JESUS, 18 de dezembro de 2018 ELVIO IBSEN BARRETO DE SOUZA COUTINHO-Juiz(a) de Direito da Vara Única da Comarca de BOM JESUS.'.lower()]
texto = tfidf.transform(texto)


# In[80]:


clf.predict(texto)


# In[1]:


####################### Testes
from joblib import load

model_stable = load('EvidenciaSentenca.joblib')


# In[38]:


saida=model_stable.predict(['Processo nº 0000001-66.2009.8.18.0042 Classe: Execução de Título Extrajudicial Exequente: ROYTON QUIMICA FARMACEUTICA LTDA Advogado(s): DOMINGOS GUSTAVO DE SOUZA(OAB/SÃO PAULO Nº 26283) Executado(a): F P RODRIGUES Advogado(s): SENTENÇA (...) Assim, impõe-se a extinção do processo por restar evidenciada a falta de utilidade e/ou necessidade na sua continuidade, fazendo desaparecer uma das condições da ação, no caso, o interesse processual. Pelo exposto, com fundamento no art. 485, III e VI do NCPC, julgo extinto o processo sem exame do mérito. Condeno a parte autora no pagamento das custas processuais. Sem condenação em honorários. Publique-se. Registre-se. Intimem-se. BOM JESUS, 18 de dezembro de 2018 ELVIO IBSEN BARRETO DE SOUZA COUTINHO-Juiz(a) de Direito da Vara Única da Comarca de BOM JESUS.'])


# In[39]:


saida


# In[5]:


saida[0]


# In[33]:


tf_train


# In[55]:


type(tf_train)


# In[81]:


from os import getenv
import pandas as pd
import numpy as np
import re
import sys
from glob import glob
from functools import partial
import datetime
import os
import gc


# In[84]:


Arquivo= '/home/leandro.rodriguez/projetos/python/sentencas/dadossentenca/SentencaCorpus'+"-"+'.csv'
arquivos = glob(Arquivo)


# In[85]:


arquivos


# In[4]:


get_ipython().system('pip show pandas')


# In[ ]:




