import requests
import folium
from folium.features import CustomIcon
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
import torch
import os
from openai import OpenAI

model_sentiment_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer_sentiment = AutoTokenizer.from_pretrained(model_sentiment_name)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(model_sentiment_name)


# 取得地點
def fetch_places(keyword, api_key, location):
    radius = 1000
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={keyword}&location={location}&radius={radius}&key={api_key}"
    res = requests.get(url)
    return res.json().get('results', [])


# 取得評論
def fetch_reviews(place_id, api_key, language='zh-TW'):
    # url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&key={api_key}"
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&language={language}&key={api_key}"
    res = requests.get(url).json()
    try:
        return [review['text'] for review in res['result']['reviews']]
    except:
        return []

# 預測情感
def predict_sentiment(texts):
    if not texts:
        return []

    # inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = tokenizer_sentiment(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]

# 計算平均情感分數
def average_sentiment_score(sentiments):
    score_map = {"Very Negative": 0, "Negative": 1, "Neutral": 2, "Positive": 3, "Very Positive": 4}
    if not sentiments:
        return "N/A", 0
    scores = [score_map[s] for s in sentiments]
    return f"{sum(scores)/len(scores):.2f}", len(scores)


# 取得評論的關鍵意見句
def extract_keywords(reviews, client, model="gpt-3.5-turbo"):

    # Join all reviews as one input
    review_text = "\n".join(reviews)

    prompt = f"""
        請從以下評論中萃取出關鍵意見句，著重於具體描述的部分（注意！每句必須在六個字以內，例如：環境乾淨、服務員態度差、菜單選項少），並以條列形式列出：
        評論：
        {review_text}
        請輸出一組關鍵意見句清單（不需要說明）：
        """

    # 呼叫 OpenAI API
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.3)

    content = response.choices[0].message.content

    # Extract each keyword sentence from the output (assuming bullet point or line breaks)
    keywords = [line.strip("•- ").strip() for line in content.splitlines() if line.strip()]
    return keywords


# def get_earliest_review_date(place_id, api_key):
#     url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&key={api_key}"
#     res = requests.get(url).json()
#     try:
#         timestamps = [review['time'] for review in res['result']['reviews']]
#         earliest = min(timestamps)
#         return datetime.utcfromtimestamp(earliest).strftime('%Y-%m-%d')
#     except:
#         return "Unknown"

def generate_map(places, api_key, keyword, location, client):
    lat, lng = map(float, location.split(','))
    m = folium.Map(location=[lat, lng], zoom_start=15)
    ...

    for place in places:
        loc = place['geometry']['location']
        name = place['name']
        place_id = place['place_id']
        reviews = fetch_reviews(place_id, api_key)

        keywords = extract_keywords(reviews, client)

        # print(f"Reviews for {name}: {reviews}")
        sentiments = predict_sentiment(reviews)
        avg_score, review_count = average_sentiment_score(sentiments)
        # summary = summarize_reviews(reviews)

        keywords_html = "<ul style='padding-left:18px; margin:5px 0;'>" + "".join(
            f"<li>{kw}</li>" for kw in keywords[:10]
        ) + "</ul>"

        popup_html = f"""
        <b>{name}</b><br>
        <b>Sentiment Score:</b> {avg_score}<br>
        <b>Keywords:</b> {keywords_html}
        """




        folium.Marker(
            location=[loc['lat'], loc['lng']],
            popup=popup_html
        ).add_to(m)

    os.makedirs("static", exist_ok=True)
    m.save("static/map_with_opening_dates.html")

