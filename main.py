from model import treinar_modelo, analisar_sentimento

modelo, vectorizer = treinar_modelo()

texto = input("Digite um texto para análise de sentimento: ")
resultado = analisar_sentimento(modelo, vectorizer, texto)

print(f"Sentimento detectado: {resultado}")
