# coding: utf-8
import nltk
from sklearn.model_selection import train_test_split
import csv
import pandas as pd


base = []

crdados = csv.reader(open("dados.csv","r"))

for rown in crdados:
    base.append(rown)


df_X = pd.DataFrame([elemento for elemento in base]) # load the dataset as a pandas data frame
df_Y = pd.DataFrame([elemento[1] for elemento in base])


a_train, a_test= train_test_split(df_X, test_size=0.33, random_state=42)


basetreinamento = a_train.values.tolist()
baseteste = a_test.values.tolist()

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
stopwordsnltk.append("vou")
#print(stopwordsnltk)

def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases

#print(removestopwords(base))

def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasessstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming

frasescomstemmingtreinamento = aplicastemmer(basetreinamento)
frasescomstemmingteste = aplicastemmer(baseteste)
#print(frasescomstemming)

def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavrastreinamento = buscapalavras(frasescomstemmingtreinamento)
palavrasteste = buscapalavras(frasescomstemmingteste)
#print(palavras)

def buscafrequencia(palavras):
    print(type(palavras))
    palavras = nltk.FreqDist(palavras)
    return palavras

frequenciatreinamento = buscafrequencia(palavrastreinamento)
frequenciateste = buscafrequencia(palavrasteste)
#print(frequencia.most_common(50))

def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicastreinamento = buscapalavrasunicas(frequenciatreinamento)
palavrasunicasteste = buscapalavrasunicas(frequenciateste)
#print(palavrasunicastreinamento)

#print(palavrasunicas)

def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicastreinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas



basecompletatreinamento = nltk.classify.apply_features(extratorpalavras, frasescomstemmingtreinamento)
basecompletateste = nltk.classify.apply_features(extratorpalavras, frasescomstemmingteste)
#print(basecompleta[15])

# constroi a tabela de probabilidade
classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)
print(classificador.labels())
print(classificador.show_most_informative_features(20))

print(nltk.classify.accuracy(classificador, basecompletateste))


erros = []
for (frase, classe) in basecompletateste:
    #print(frase)
    #print(classe)
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))
#for (classe, resultado, frase) in erros:
#    print(classe, resultado, frase)

print(erros)


from nltk.metrics import ConfusionMatrix
esperado = []
previsto = []
for (frase, classe) in basecompletateste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)


matriz = ConfusionMatrix(esperado, previsto)
print(matriz)


teste = 'me mate'
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))
#print(testestemming)

novo = extratorpalavras(testestemming)
#print(novo)

print(classificador.classify(novo))
distribuicao = classificador.prob_classify(novo)
#for classe in distribuicao.samples():
#    print("%s: %f" % (classe, distribuicao.prob(classe)))








