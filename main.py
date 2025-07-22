from PairGenerator import generateClaim, searchClaim, createPairs, summarizePairs
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

modelName = 'roberta-large-mnli'
tokenizer = AutoTokenizer.from_pretrained(modelName)
bertModel = AutoModelForSequenceClassification.from_pretrained(modelName)


def classifyPair(premise, hypothesis):
    inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = bertModel(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0].tolist()
    labels = ['contradiction', 'neutral', 'entailment']
    maxIdx = torch.argmax(torch.tensor(probs)).item()
    return labels[maxIdx], probs


def runVerificationPipeline(userInput):
    print(f"\nUser Input:\n{userInput}\n")

    claim = generateClaim(userInput)
    print(f"Generated Claim:\n{claim}\n")

    searchResults = searchClaim(claim)
    print("Search Results:")
    for i, r in enumerate(searchResults, 1):
        print(f"{i}. {r['title']}\n   {r['snippet']}\n   {r['link']}\n")

    inputPairs = createPairs(claim, searchResults)
    print("Premise-Hypothesis Pairs & Predictions:")
    for i, (p, h) in enumerate(inputPairs, 1):
        label, probs = classifyPair(p, h)
        print(f"\nPair {i}:")
        print(f"Premise: {p}")
        print(f"Hypothesis: {h}")
        print(f"Prediction: {label}")
        print(f"Probabilities: {probs}")

    summary = summarizePairs(inputPairs)
    print("\nSummary:\n", summary)



runVerificationPipeline("Drinking celery juice every day COMPLETELY protects you from catching any colds! #HealthFacts")