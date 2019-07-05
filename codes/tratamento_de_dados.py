'''
Machine Learning para Analise de Dados
CESAR Scholl 

Projeto de analise de credito
Recife, 2019
@Claudio Alves Monteiro
'''

#===============================#
# modules, settings and data
#==============================#

# import modules
import os
import pandas as pd
from pyspark.sql import functions as SF
import pyspark.sql.types as ST

# paths to spark and python3
os.environ['PYSPARK_SUBMIT_ARGS'] = '--executor-memory 1G pyspark-shell'
os.environ["SPARK_HOME"] = "/home/pacha/spark"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python3"

# execute PYSPARK port: 127.0.0.1:4040
exec(open('/home/pacha/spark/python/pyspark/shell.py').read())

# import data
df = spark.read.csv('data/BASE_JUR001.csv',
                    sep='\t',
                    encoding='utf-8',
                    header=True,
                    inferSchema=False)

dt = spark.read.csv('data/Base_Des-TRN.txt',
                    sep='\t',
                    encoding='utf-8',
                    header=True,
                    inferSchema=False)

# set display
pd.set_option('display.max_columns', None)

#=========================#
# descriptive statistics
#========================#

df.groupby('NATUREZA_JURIDICA').count().show(100)

df.groupby('ATIVIDADE_PRINCIPAL').count().show(100)

df.groupby('ATIVIDADES_SECUNDARIAS').count().show(100)

df.groupby('TIPO').count().show(100)

df.groupby('CARGO').count().show(100)

#===========================================#
# data wrangling and feature engineering
#========================================#

df.show(5)
dt.show(5)

#------ create dataset without duplicates
cnpj = df.dropDuplicates(['CNPJ'])

#------ create feature
features = cnpj.select('CNPJ', 'CAPITAL_SOCIAL')

#------ tratar inicio do CNPJ sem o 0
@SF.udf('string')
def fillCNPJ(value):
    aux = '00000000000000' + value
    return aux[len(value):]

features = features.withColumn('CNPJ', fillCNPJ('CNPJ'))

#--------- mineracao de texto do nome da empresa [palavras comuns]
# https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml import Pipeline

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="NOME_EMPRESA", outputCol="nomes", pattern="\\W")

# stop words
add_stopwords =  'de a o que e do da em um para é com não uma os no se na por mais as dos como mas foi ao ele das tem à seu sua ou ser quando muito há nos já está eu'.split(' ')
stopwordsRemover = StopWordsRemover(inputCol="nomes", outputCol="filtered").setStopWords(add_stopwords)

# apply pipeline
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover])

# Fit the pipeline to training dataset
pipelineFit = pipeline.fit(cnpj)
dataset = pipelineFit.transform(cnpj)
dataset.show(5)

# count and filter words most commom 
cnpj = cnpj.withColumn('empresa', SF.lower(SF.col('NOME_EMPRESA')))

wordcount = cnpj.withColumn('class_words', SF.explode(SF.split(SF.col('empresa'), ' ')))\
                            .groupBy('class_words')\
                            .count()\
                            .sort('count', ascending=False)

# filter count and STOPWORD
wordcount = wordcount.filter(wordcount['count'] > 5000 )
wordcount.show()

# create LIST OF WORDS MOST COMMOM
x = wordcount.select("class_words").rdd.flatMap(lambda x: x).collect()
lst = SF.array(x)
import numpy as np 
#x = np.array(x)
x.append('petiscos')

'''
# create column of word class 
@SF.udf
def countInLists(feature_list): # GAMBIARRA WORKING:
    label_list = ['ltda', 'de', 'e', 'restaurante', 'eireli', 'comercio', 'alimentos', 'da', 'silva', '&', 'lanchonete',
     'pamonharia', 'lachonete', 'lanchonete', 'bar', 'petiscos']
    class_words = []
    for label in label_list:
        if label in feature_list:
            class_words.append(label)
    return class_words

dataset.withColumn("testando", countInLists('filtered')).show(5)
'''

from functools import partial


test = dataset.withColumn('classe_palavras', countInLists('filtered', x))
test.show()


#-------------- mineracao de texto do nome fantasia [palavras comuns]

# criar colunas 0-1 com NATUREZA_JURIDICA 

# criar colunas 0-1 com ATIVIDADE_PRINCIPAL > 1

# criar colunas 0-1 com ATIVIDADES_SECUNDARIAS > 1

# criar colunas 0-1 com TIPO 

# mineracao de texto do COMPLEMENTO [LOJA, AP...] 

# quantos telefones possui (TELEFONE_1 e TELEFONE_2)

# possui EMAIL 0-1

# tempo de existencia data hoje-ABERTURA

# criar colunas 0-1 com TIPO 

# contar numero de socios por CNPJ 

# mineracao de texto do NOME_SOCIO [prop. de mulher do total] 

#--- explorar dados por CNPJ

#--- explorar dados por CEP

#--- explorardados por MUNICIPIO

