import google.generativeai as genai
from serpapi import GoogleSearch


# Gemini (creating claim)
genai.configure(api_key="GEMINI_API_KEY")

def generateClaim(userText):
    prompt = (
        f'Extract a short, clear factual claim from the following text: "{userText}"\n'
        "Only output the claim itself with no extra explanation or introductory words."
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    claimText = response.text
    return claimText.strip()



# Serp
def searchClaim(claim):
    params = {
        'engine': 'google',
        'q': claim,
        'api_key': '"SERP_API_KEY"',
        'num': 5,
        'hl': 'en',
        'gl': 'us' 
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    top_results = results.get("organic_results", [])
    return [
        {"title": r["title"], "snippet": r["snippet"], "link": r["link"]}
        for r in top_results
    ]



# Creating Pairs
def createPairs(claim, searchResults):
    pairs = []

    for result in searchResults:
        premise = result['snippet']
        hypothesis = claim
        pairs.append((premise, hypothesis))
    return pairs




# Main Pipeline


userCaption = "OMG I heard drinking celery juice every day makes you totally immune to colds! 😱🥒"     # Test input

# Generating Claim (Gemini)
claim = generateClaim(userCaption)
print("Generated claim:", claim)

# Supporting snippets (SERP)
searchResults = searchClaim(claim)

# Combine into pairs
inputPairs = createPairs(claim, searchResults)

print("\nPairs for BERT input:")
for i, (premise, hypothesis) in enumerate(inputPairs, 1):
    print(f"Pair {i}:")
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}\n")
