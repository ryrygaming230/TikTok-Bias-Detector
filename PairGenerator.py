import google.generativeai as genai
from serpapi import GoogleSearch

genai.configure(api_key="AIzaSyBRY7TFhs-8NsaqaDiOXwbR91uuFL5jLLM")
geminiModel = genai.GenerativeModel("gemini-1.5-flash")

def generateClaim(userText):
    prompt = (
        f'Extract a short, clear factual claim from the following text: "{userText}"\n'
        "Only output the claim itself with no extra explanation or introductory words."
    )
    response = geminiModel.generate_content(prompt)
    claimText = response.text
    return claimText.strip()

def searchClaim(claim):
    params = {
        'engine': 'google',
        'q': claim,
        'api_key': 'ebc56185e5431e2048bb3b5ef89147055568a754da5fce1335070eadf21f3587',
        'num': 5,
        'hl': 'en',
        'gl': 'us'
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    topResults = results.get("organic_results", [])
    return [
        {"title": r["title"], "snippet": r["snippet"], "link": r["link"]}
        for r in topResults
    ]

def createPairs(claim, searchResults):
    pairs = []
    for result in searchResults:
        premise = result['snippet']
        hypothesis = claim
        pairs.append((premise, hypothesis))
    return pairs

def summarizePairs(pairs):
    joined = "\n\n".join([f"Premise: {p}\nHypothesis: {h}" for p, h in pairs])
    prompt = (
        f"Summarize the key takeaways or common points from the following claim-verification pairs:\n\n{joined}\n\n"
        "Only output a brief factual summary. Use simple, everyday language that's easy for regular people to understand when reading casually."
    )
    response = geminiModel.generate_content(prompt)
    return response.text.strip()
