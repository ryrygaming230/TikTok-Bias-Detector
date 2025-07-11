import google.generativeai as genai


def generateClaim(final):
    generatedPair = (
        f'Use the following text and generate a summary: "{generatedPair}"\n'

    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(final)
    claimText = response.text
    return claimText.strip()