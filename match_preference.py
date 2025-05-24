from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
import time

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
    if not texts:
        return []

    prompt = (
        "請判斷以下每個句子的情感傾向。每行格式為：<句子> -> <情感分類>。\n"
        "情感分類請選擇：Very Negative, Negative, Neutral, Positive, Very Positive。\n\n"
    )
    prompt += "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一個情感分析助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        output = response.choices[0].message.content.strip()

        sentiments = []
        for line in output.splitlines():
            if "->" in line:
                try:
                    _, sentiment = line.split("->", 1)
                    sentiments.append(sentiment.strip())
                except:
                    sentiments.append("Neutral")
            else:
                sentiments.append("Neutral")
        return sentiments

    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return ["Neutral"] * len(texts)

def batch_sentiment_match(preferences, keywords):
    """
    For each preference-keyword pair, determine sentiment agreement.
    Returns a dictionary of (pref, keyword) -> True/False
    """
    pairs = [(p, k) for p in preferences for k in keywords]
    flat_texts = [p for p, k in pairs] + [k for p, k in pairs]
    flat_sentiments = predict_sentiment(flat_texts)

    n = len(pairs)
    pref_sents = flat_sentiments[:n]
    key_sents = flat_sentiments[n:]

    polarity_map = {
        "Very Negative": -2,
        "Negative": -1,
        "Neutral": 0,
        "Positive": 1,
        "Very Positive": 2
    }

    match_dict = {}
    for i, (p, k) in enumerate(pairs):
        ps = polarity_map.get(pref_sents[i], 0)
        ks = polarity_map.get(key_sents[i], 0)
        match_dict[(p, k)] = not (ps * ks < 0)
    return match_dict

def match_preferences(preferences, keywords, threshold=0.75):
    pref_embeds = [get_embedding(p) for p in preferences]
    key_embeds = [get_embedding(k) for k in keywords]

    sentiment_agreement = batch_sentiment_match(preferences, keywords)

    best_matches = []
    for i, p_emb in enumerate(pref_embeds):
        best_match = None
        best_score = 0
        for j, k_emb in enumerate(key_embeds):
            sim = cosine_similarity(p_emb, k_emb)
            if sim >= threshold:
                if sentiment_agreement.get((preferences[i], keywords[j]), False):
                    if sim > best_score:
                        best_match = (preferences[i], keywords[j], sim)
                        best_score = sim
        if best_match:
            best_matches.append(best_match)
            print(f"Best Match for {preferences[i]}: {best_match[0]} with {best_match[1]} - Similarity: {best_match[2]:.4f}")
    return best_matches

if __name__ == "__main__":
    preferences = ["服務態度好", "環境乾淨"]
    keywords = ["服務態度不佳", "環境很髒", "服務態度很好", "乾淨整潔"]

    start_time = time.time()
    match_score = match_preferences(preferences, keywords)
    end_time = time.time()

    print(f"Final Match Score: {match_score}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
