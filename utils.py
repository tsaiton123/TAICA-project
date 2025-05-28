import requests
import folium
from folium.features import CustomIcon
from folium import Html, Popup
from datetime import datetime
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
# import torch
import os
from openai import OpenAI
from branca.element import Figure
from match_preference import match_preferences
from concurrent.futures import ThreadPoolExecutor, as_completed


# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

import json
from db import db
from models import PlaceCache

def relation_score(preference, keywords):
    matches = match_preferences(preference, keywords)
    if not matches:
        return ("", "", 0.0)
    best_keyword, matched_text, score = matches[0]  # assume only the first preference
    return (best_keyword, matched_text, score)


def get_or_compute(place, api_key, client, preference):
    # 1. Try cache
    pc = PlaceCache.query.filter_by(place_id=place['place_id']).first()
    if pc:
        keywords = json.loads(pc.keywords_json)
        relation = relation_score(preference, keywords)
        return json.loads(pc.keywords_json), relation

    # 2. Not cached → run inference
    reviews    = fetch_reviews(place['place_id'], api_key)
    keywords   = extract_keywords(reviews, client)
    relation = relation_score(preference, keywords)

    # 3. Save to cache
    pc = PlaceCache(
        place_id      = place['place_id'],
        name          = place['name'],
        keywords_json = json.dumps(keywords)
    )
    db.session.add(pc)
    db.session.commit()

    # return keywords, float(r_score) if r_score != "N/A" else 0
    return keywords, relation



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


# 取得評論的關鍵意見句
def extract_keywords(reviews, client, model="gpt-3.5-turbo"):

    # Join all reviews as one input
    review_text = "\n".join(reviews)

    prompt = f"""
            你是一個「字數管理員」：請從下列評論中嚴格萃取「最多六個中文字」的關鍵意見句，每句不得超過六個中文字（不含標點符號），超過請自動重寫為等義的六字內短句，或捨棄。輸出格式為純粹的條列清單，每行開頭用「- 」，不加其他文字。  
            評論：
            {review_text}

            範例正確格式：
            - 環境乾淨
            - 服務態度差
            - 菜單選擇少

            請開始：
            """

    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.3)

    content = response.choices[0].message.content

    # Extract each keyword sentence from the output (assuming bullet point or line breaks)
    keywords = [line.strip("•- ").strip() for line in content.splitlines() if line.strip()]
    return keywords


# def generate_map(places, api_key, preference, location, client):
#     lat, lng = map(float, location.split(','))
#     m = folium.Map(location=[lat, lng], zoom_start=15)

#     best_place = None
#     best_score = -1
#     place_infos = []

#     for place in places:
#         loc = place['geometry']['location']
#         name = place['name']
#         place_id = place['place_id']

#         keywords, (best_keyword, matched_text, score) = get_or_compute(place, api_key, client, preference)

#         place_infos.append((place, loc, name, keywords, best_keyword, matched_text, score))

#         if score > best_score:
#             best_score = score
#             best_place = place_id  # or store full data as best_place_info = (place, ...)

#     for (place, loc, name, keywords, best_keyword, matched_text, score) in place_infos:
#         is_best = (place['place_id'] == best_place)

#         popup_html = f"""
#         <b>{name}</b><br>
#         <b>Best Match:</b> {matched_text} ({score:.2f})<br>
#         <b>Keywords:</b>
#         <ul style='padding-left:18px; margin:5px 0;'>
#             {''.join(f"<li>{kw}</li>" for kw in keywords[:10])}
#         </ul>
#         """

#         html = Html(popup_html, script=True)
#         popup = Popup(html, max_width=300)

#         marker_color = "red" if is_best else "blue"
#         folium.Marker(
#             location=[loc['lat'], loc['lng']],
#             popup=popup,
#             icon=folium.Icon(color=marker_color)
#         ).add_to(m)


#     os.makedirs("static", exist_ok=True)
#     m.save("static/map_with_opening_dates.html")

def generate_map(places, api_key, preference, location, client):
    lat, lng = map(float, location.split(','))
    m = folium.Map(location=[lat, lng], zoom_start=15)

    best_place = None
    best_score = -1
    place_infos = []

    def process_place(place):
        loc = place['geometry']['location']
        name = place['name']
        place_id = place['place_id']
        try:
            keywords, (best_keyword, matched_text, score) = get_or_compute(place, api_key, client, preference)
            return (place, loc, name, keywords, best_keyword, matched_text, score)
        except Exception as e:
            print(f"Error processing {place_id}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_place, place) for place in places]

        for future in as_completed(futures):
            result = future.result()
            if result:
                place, loc, name, keywords, best_keyword, matched_text, score = result
                place_infos.append(result)
                if score > best_score:
                    best_score = score
                    best_place = place['place_id']

    # Add markers to map
    for (place, loc, name, keywords, best_keyword, matched_text, score) in place_infos:
        is_best = (place['place_id'] == best_place)

        popup_html = f"""
        <b>{name}</b><br>
        <b>Best Match:</b> {matched_text} ({score:.2f})<br>
        <b>Keywords:</b>
        <ul style='padding-left:18px; margin:5px 0;'>{''.join(f"<li>{kw}</li>" for kw in keywords[:10])}</ul>
        """
        html = Html(popup_html, script=True)
        popup = Popup(html, max_width=300)
        marker_color = "red" if is_best else "blue"

        folium.Marker(
            location=[loc['lat'], loc['lng']],
            popup=popup,
            icon=folium.Icon(color=marker_color)
        ).add_to(m)

    os.makedirs("static", exist_ok=True)
    m.save("static/map_with_opening_dates.html")
