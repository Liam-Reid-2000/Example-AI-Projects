from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
  "At Enzai Technologies, we",
  max_length=100,
  num_return_sequences=1,
)


print(res)