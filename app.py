from flask import Flask, render_template, request
from utils import fetch_places, generate_map
from openai import OpenAI
from dotenv import load_dotenv
import os
import folium

from shapely.geometry import Point, box
import geopandas as gpd
import folium

def add_flashlight_with_geopandas(map_obj, lat, lng, radius_meters=500):
    from shapely.geometry import Point, box
    import geopandas as gpd
    import folium

    user_point = Point(float(lng), float(lat))
    num_rings = 100
    base_radius = radius_meters
    max_radius = base_radius * (1 + 0.6)

    # Full dark outer box
    world_box = box(-180, -90, 180, 90)

    # Create rings with decreasing opacity toward center
    for i in range(num_rings):
        # Outermost ring is fully dark
        inner_radius = base_radius * (i / num_rings)
        outer_radius = base_radius * ((i + 1) / num_rings)

        inner_deg = inner_radius / 111_000
        outer_deg = outer_radius / 111_000

        outer_buffer = user_point.buffer(outer_deg)
        inner_buffer = user_point.buffer(inner_deg)
        ring = outer_buffer.difference(inner_buffer)

        mask = world_box.difference(inner_buffer)
        visible_hole = mask.difference(ring)

        gdf = gpd.GeoDataFrame(geometry=[visible_hole])
        opacity = 0.9 * ((i + 1) / num_rings)  # more opaque outward

        folium.GeoJson(
            data=gdf.__geo_interface__,
            style_function=lambda x, op=opacity: {
                'fillColor': 'black',
                'color': 'black',
                'fillOpacity': op,
                'weight': 0,
            }
        ).add_to(map_obj)







load_dotenv()
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

    # === Landing map with flashlight effect ===
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    if lat and lng:
        lat_f, lng_f = float(lat), float(lng)
        m = folium.Map(location=[lat_f, lng_f], zoom_start=17, tiles="OpenStreetMap")
        folium.Marker([lat_f, lng_f], popup="You").add_to(m)
        add_flashlight_with_geopandas(m, lat_f, lng_f)

        os.makedirs("static", exist_ok=True)
        m.save("static/map_with_opening_dates.html")
        return render_template('index.html', keyword=None)

    return render_template('index.html', keyword=None)

if __name__ == '__main__':
    app.run(debug=True)
