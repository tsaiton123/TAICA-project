from flask import Flask, render_template, request
from utils import fetch_places, get_earliest_review_date, generate_map
import dotenv
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env

openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        lat = request.form.get('lat', '25.033964')
        lng = request.form.get('lng', '121.564468')
        location = f"{lat},{lng}"
        keyword = request.form['keyword']


        places = fetch_places(keyword, google_api_key, location)
        client = OpenAI(api_key=openai_api_key)
        generate_map(places, google_api_key, keyword, location, client)
        return render_template('index.html', keyword=keyword)

    return render_template('index.html', keyword=None)

if __name__ == '__main__':
    app.run(debug=True)
