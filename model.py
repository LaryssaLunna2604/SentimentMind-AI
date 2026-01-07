from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def treinar_modelo():
    textos = [
        "eu amo este produto",
        "isso é horrível",
        "serviço excelente",
        "péssima experiência",
        "muito bom",
        "não gostei"
    ]

    sentimentos = ["positivo", "negativo", "positivo", "negativo", "positivo", "negativo"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(textos)

    modelo = MultinomialNB()
    modelo.fit(X, sentimentos)

    return modelo, vectorizer

def analisar_sentimento(modelo, vectorizer, texto):
    vetor = vectorizer.transform([texto])
    return modelo.predict(vetor)[0]
