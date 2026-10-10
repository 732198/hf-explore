import time

from transformers import pipeline

MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

headlines = [
    "Shares of the vineyard sensor maker jumped 18% after it doubled its quarterly revenue guidance.",
    "The retailer will close 40 stores and cut 1,200 jobs after a second straight quarterly loss.",
    "The company will hold its annual shareholder meeting in Boston on March 12.",
]

start = time.perf_counter()
classifier = pipeline("text-classification", model=MODEL)
print(f"Model loaded in {time.perf_counter() - start:.1f} seconds\n")

start = time.perf_counter()
results = classifier(headlines)
print(f"Classified {len(headlines)} headlines in {time.perf_counter() - start:.2f} seconds\n")

for headline, result in zip(headlines, results):
    print(headline)
    print(f"  -> {result['label']} ({result['score']:.1%} confidence)\n")
