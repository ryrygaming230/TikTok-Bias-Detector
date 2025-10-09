# app.py

# --- Imports ---
import google.generativeai as genai
from serpapi import GoogleSearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Gemini and HuggingFace Model Initialization ---
# NOTE: keep keys in env vars for real deployments.
GEMINI_API_KEY = "YOUR_API_KEY"
SERPAPI_KEY = "YOUR_API_KEY"

genai.configure(api_key=GEMINI_API_KEY)
geminiModel = genai.GenerativeModel("gemini-2.5-flash")

modelName = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(modelName)
bertModel = AutoModelForSequenceClassification.from_pretrained(modelName)

# --- Core Functions ---
def classifyPair(premise, hypothesis):
    inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = bertModel(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0].tolist()
    labels = ["contradiction", "neutral", "entailment"]
    maxIdx = int(torch.argmax(torch.tensor(probs)))
    return labels[maxIdx], probs

def generateClaim(userText):
    prompt = (
        f'Extract a short, clear factual claim from the following text: "{userText}"\n'
        "Only output the claim itself with no extra explanation or introductory words."
    )
    response = geminiModel.generate_content(prompt)
    return (response.text or "").strip()

def searchClaim(claim):
    params = {"engine": "google", "q": claim, "api_key": SERPAPI_KEY, "num": 5, "hl": "en", "gl": "us"}
    search = GoogleSearch(params)
    results = search.get_dict()
    topResults = results.get("organic_results", []) or []
    cleaned = []
    for r in topResults:
        title = r.get("title") or "Source"
        snippet = r.get("snippet") or ""
        link = r.get("link") or "#"
        cleaned.append({"title": title, "snippet": snippet, "link": link})
    return cleaned

def rephraseSnippet(snippet):
    if not snippet:
        return ""
    prompt = f"Turn this search snippet into a clear, factual sentence:\nSnippet: {snippet}"
    response = geminiModel.generate_content(prompt)
    return (response.text or "").strip()

def createPairs(claim, searchResults):
    pairs = []
    for result in searchResults:
        premise = rephraseSnippet(result.get("snippet", ""))
        hypothesis = claim
        if premise:
            pairs.append((premise, hypothesis))
    return pairs

def summarizePairs(pairs):
    if not pairs:
        return "We couldn’t find enough high-quality evidence to make a clear call."
    joined = "\n\n".join([f"Premise: {p}\nHypothesis: {h}" for p, h in pairs])
    prompt = (
        "Summarize the key takeaways or common points from the following claim-verification pairs:\n\n"
        f"{joined}\n\n"
        "Only output a brief factual summary in simple, teen-friendly language."
    )
    response = geminiModel.generate_content(prompt)
    return (response.text or "").strip()

# --- FastAPI App ---
app = FastAPI(
    title="Claim Verification API",
    description="Runs a claim verification pipeline using Gemini and NLI models.",
    version="1.0.0",
)

# CORS for local dev & static files opened in the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimInput(BaseModel):
    userInput: str

@app.get("/")
def root():
    return {"ok": True, "message": "FactTok API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/verify-claim/")
async def verify_claim_endpoint(input_data: ClaimInput):
    """
    Full claim verification pipeline.
    """
    userInput = input_data.userInput.strip()
    if not userInput:
        return {
            "userInput": "",
            "generatedClaim": "",
            "summary": "Please provide a URL, caption, or claim.",
            "verificationResults": [],
            "sourceSearchResults": [],
        }

    # 1) Claim from user input (URL or text)
    claim = generateClaim(userInput) or userInput  # fall back to raw input if extractor returns empty

    # 2) Search for evidence
    searchResults = searchClaim(claim)

    # 3) Build NLI pairs & classify
    inputPairs = createPairs(claim, searchResults)
    classified_results = []
    all_pairs = []
    for p, h in inputPairs:
        label, probs = classifyPair(p, h)
        pair_data = {
            "premise": p,
            "hypothesis": h,
            "prediction": label,
            "probabilities": {
                "contradiction": probs[0],
                "neutral": probs[1],
                "entailment": probs[2],
            },
        }
        all_pairs.append((p, h))
        classified_results.append(pair_data)

    # 4) Summary
    summary = summarizePairs(all_pairs)

    # 5) Response
    return {
        "userInput": userInput,
        "generatedClaim": claim,
        "summary": summary,
        "verificationResults": classified_results,
        "sourceSearchResults": searchResults,
    }
