# Você recebeu um arquivo CSV com 100 avaliações de clientes sobre um smartphone. Agora, seu desafio
# é treinar um classificador de sentimentos usando esse conjunto de dados para prever se uma nova
# review é positiva, negativa ou neutra.
# Etapas da Atividade
# 1.Carregar o dataset
# • Ler o arquivo reviews_classificados.csv usando pandas. ok
# 2.Pré-processamento
# • Tokenizar os textos ok
# • Colocar tudo em minúsculas ok
# • Remover stopwords ok
# • Realizar stemming ou lemmatization ?
# 3.Criação do dataset de treino
# • Transformar os dados em uma estrutura de tuplas: (dicionario_de_palavras, sentimento)
# 4.Treinamento do classificador
# • Usar nltk.NaiveBayesClassifier para treinar o modelo

import pandas as pd
import string
import nltk
nltk.download(['punkt', 'punkt_tab', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier, classify

LANGUAGE = 'portuguese'

def crie_dataset_de_treino(arquivo):
    stop_words = set(stopwords.words(LANGUAGE))
    dataset = pd.read_csv(arquivo)
    dataset_treino = []

    for linha in dataset.iterrows():
        review = linha[1]['Reviews']
        sentimento = linha[1]['Sentimento']
        review_sem_pontuacao = review.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(review_sem_pontuacao, language=LANGUAGE)
        tokens_minusculo = [token.lower() for token in tokens]
        tokens_filtrados = [token for token in tokens_minusculo if token not in stop_words]
        dataset_treino.append(({token: True for token in tokens_filtrados}, sentimento))

    return dataset_treino

if __name__ == '__main__':
    dataset_treino = crie_dataset_de_treino('reviews_classificados.csv')
    treino = dataset_treino[:70]
    teste = dataset_treino[70:]
    classificador = NaiveBayesClassifier.train(treino)
    print(f'A acurácia é de: {classify.accuracy(classificador, teste)}')

    teste_aula_palavras = teste[0][0]
    teste_aula_sentimento = teste[0][1]

    print(f'O sentimento da review é: {classificador.classify(teste_aula_palavras)}')
    print(f'O sentimento esperado é: {teste_aula_sentimento}')
