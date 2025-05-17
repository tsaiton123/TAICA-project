# app.py
import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
import folium
from openai import OpenAI
from shapely.geometry import Point, box
import geopandas as gpd

from db import db
from models import PlaceCache
from utils import fetch_places, generate_map

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
database_url    = os.getenv("DATABASE_URL", "sqlite:///cache.db")

def create_app():
    app = Flask(__name__)
    # --- Database config & init ---
    app.config["SQLALCHEMY_DATABASE_URI"]        = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    # Create tables
    with app.app_context():
        db.create_all()

    # --- Flashlight helper ---
    def add_flashlight_with_geopandas(map_obj, lat, lng, radius_meters=500):
        user_point = Point(float(lng), float(lat))
        world_box  = box(-180, -90, 180, 90)
        num_rings  = 100
        for i in range(num_rings):
            inner_radius = radius_meters * (i / num_rings)
            outer_radius = radius_meters * ((i + 1) / num_rings)
            inner_deg = inner_radius / 111_000
            outer_deg = outer_radius / 111_000

            inner_buffer = user_point.buffer(inner_deg)
            outer_buffer = user_point.buffer(outer_deg)
            ring         = outer_buffer.difference(inner_buffer)

            # mask out hole + ring
            mask         = world_box.difference(inner_buffer)
            visible_area = mask.difference(ring)

            gdf     = gpd.GeoDataFrame(geometry=[visible_area])
            opacity = 0.9 * ((i + 1) / num_rings)
            folium.GeoJson(
                data=gdf.__geo_interface__,
                style_function=lambda feature, op=opacity: {
                    "fillColor":  "black",
                    "color":      "black",
                    "fillOpacity": op,
                    "weight":     0,
                }
            ).add_to(map_obj)

    # --- Routes ---
    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            lat     = request.form.get("lat", "25.033964")
            lng     = request.form.get("lng", "121.564468")
            location = f"{lat},{lng}"
            keyword  = request.form["keyword"]

            places = fetch_places(keyword, google_api_key, location)
            client = OpenAI(api_key=openai_api_key)
            generate_map(places, google_api_key, keyword, location, client)
            return render_template("index.html", keyword=keyword)


        # landing map with flashlight
        lat = request.args.get("lat")
        lng = request.args.get("lng")
        if lat and lng:
            lat_f, lng_f = float(lat), float(lng)
            m = folium.Map(location=[lat_f, lng_f], zoom_start=17)
            folium.Marker([lat_f, lng_f], popup="You").add_to(m)
            add_flashlight_with_geopandas(m, lat_f, lng_f)
            os.makedirs("static", exist_ok=True)
            m.save("static/map_with_opening_dates.html")
            return render_template("index.html", keyword=None)

        return render_template("index.html", keyword=None)

    return app

# if __name__ == "__main__":
#     create_app().run(debug=True)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
