# from openai import OpenAI
# import numpy as np
# import os
# from dotenv import load_dotenv

# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=openai_api_key)

# def get_embedding(text):
#     response = client.embeddings.create(
#         model="text-embedding-ada-002",
#         input=text
#     )
#     return response.data[0].embedding

# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def match_preferences(preferences, keywords, threshold=0.75):
#     pref_embeds = [get_embedding(p) for p in preferences]
#     key_embeds = [get_embedding(k) for k in keywords]

#     matches = []
#     match_scores = []
#     for i, p_emb in enumerate(pref_embeds):
#         score = 0
#         for j, k_emb in enumerate(key_embeds):
#             sim = cosine_similarity(p_emb, k_emb)
#             if sim >= threshold:
#                 matches.append((preferences[i], keywords[j], sim))
#                 print(f"Match: {preferences[i]} with {keywords[j]} - Similarity: {sim:.4f}")
#                 score += sim
#         if score > 0:
#             match_scores.append((score / len(key_embeds)))
#     return match_scores


# if __name__ == "__main__":
#     preferences = ["beach", "mountain", "city"]
#     keywords = ["sunny beach", "snowy mountain", "urban city"]
#     matches = match_preferences(preferences, keywords)
#     for match in matches:
#         print(f"Match: {match[0]} with {match[1]} - Similarity: {match[2]:.4f}")


from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def predict_sentiment(texts):
    # texts: list of strings
    if not texts:
        return []
    messages = [{"role": "user", "content": t} for t in texts]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Classify sentiment as Very Negative, Negative, Neutral, Positive, or Very Positive."}] + messages,
        temperature=0,
        n=1
    )
    # Extract response for each input text (assumes single response with all sentiments concatenated)
    # Since OpenAI chat API does not support batch natively, better to call individually or use another approach
    # For demo: call individually below
    sentiments = []
    for t in texts:
        prompt = f"請判斷以下句子的情感傾向：{t}。回答為：Very Negative, Negative, Neutral, Positive, 或 Very Positive。"
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        sentiment = resp.choices[0].message.content.strip()
        sentiments.append(sentiment)
    return sentiments

def sentiment_match(phrase1, phrase2):
    sentiments = predict_sentiment([phrase1, phrase2])
    polarity_map = {
        "Very Negative": -2,
        "Negative": -1,
        "Neutral": 0,
        "Positive": 1,
        "Very Positive": 2
    }
    p1 = polarity_map.get(sentiments[0], 0)
    p2 = polarity_map.get(sentiments[1], 0)
    # Reject if one positive and other negative
    if p1 * p2 < 0:
        return False
    return True

def match_preferences(preferences, keywords, threshold=0.75):
    pref_embeds = [get_embedding(p) for p in preferences]
    key_embeds = [get_embedding(k) for k in keywords]

    # matches = []
    best_matches = []
    for i, p_emb in enumerate(pref_embeds):
        best_match = None
        best_score = 0
        for j, k_emb in enumerate(key_embeds):
            sim = cosine_similarity(p_emb, k_emb)
            if sim >= threshold:
                # Check sentiment agreement before accepting match
                if sentiment_match(preferences[i], keywords[j]):
                    # matches.append((preferences[i], keywords[j], sim))
                    if sim > best_score:
                        best_match = (preferences[i], keywords[j], sim)
                        best_score = sim
                    # print(f"Match: {preferences[i]} with {keywords[j]} - Similarity: {sim:.4f}")
                else:
                    # print(f"Rejected (sentiment mismatch): {preferences[i]} with {keywords[j]} - Similarity: {sim:.4f}")
                    pass

        if best_match:
            best_matches.append(best_match)
            print(f"Best Match for {preferences[i]}: {best_match[0]} with {best_match[1]} - Similarity: {best_match[2]:.4f}")
    return best_matches[0][2]


if __name__ == "__main__":
    preferences = ["服務態度好", "環境乾淨"]
    keywords = ["服務態度不佳", "環境很髒", "服務態度很好", "乾淨整潔"]
    # matches, best_matches = match_preferences(preferences, keywords)
    # for pref, key, score in best_matches:
    #     print(f"Final Match: '{pref}' with '{key}' - Similarity: {score:.4f}")
    natch_score = match_preferences(preferences, keywords)
    print(f"Final Match Score: {natch_score:.4f}")