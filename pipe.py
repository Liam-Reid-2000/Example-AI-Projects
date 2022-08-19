from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("Hello, my name is Liam")

print(res)