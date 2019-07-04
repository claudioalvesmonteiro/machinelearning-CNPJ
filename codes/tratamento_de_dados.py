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
from time import time

# paths to spark and python3
os.environ['PYSPARK_SUBMIT_ARGS'] = '--executor-memory 1G pyspark-shell'
os.environ["SPARK_HOME"] = "/home/pacha/spark"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python3"

# execute PYSPARK port: 127.0.0.1:4040
exec(open('/home/pacha/spark/python/pyspark/shell.py').read())

# import data
df = spark.read.csv('BASE_JUR001.csv',
                    sep='\t',
                    encoding='utf-8',
                    header=True,
                    inferSchema=False)

dt = spark.read.csv('Base_Des-TRN.txt',
                    sep='\t',
                    encoding='utf-8',
                    header=True,
                    inferSchema=False)

# set display
pandas.set_option('display.max_columns', None)

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

# create feature
features = df.select('CNPJ', 'CAPITAL_SOCIAL')

# mineracao de texto do nome da empresa [palavras comuns]

# mineracao de texto do nome fantasia [palavras comuns]

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

