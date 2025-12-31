from flask import Flask, request, jsonify
from flask_cors import CORS
from accessibility import * 

app = Flask(__name__)
CORS(app) 

@app.route("/calculate_accessibility", methods=["POST"])
def calculate_accessibility():
    """
    API endpoint to calculate accessibility scores.
    Expects JSON input with time_limit and poi_weights.
    """
    try:
        # Parse request JSON
        data = request.json
        time_limit = data.get("time_limit", 30)  # Default 30 minutes
        poi_weights = data.get("poi_weights", {})  # Default empty weights

        # Validate inputs
        if not isinstance(time_limit, int) or time_limit < 5 or time_limit > 60:
            return jsonify({"error": "Invalid time_limit. Must be between 5 and 60."}), 400

        if not isinstance(poi_weights, dict):
            return jsonify({"error": "Invalid poi_weights. Must be a dictionary."}), 400

        # Perform accessibility calculation
        result_df = run_acc(
            poi_gdf,
            centroid_points,
            walking_graph,
            subway_graph,
            subway_walk_node_mapper,
            walk_subway_node_mapper,
            centroid_station_travel_time,
            poi_weights,
            time_limit=time_limit * 60
        )

        # Create reachable pois
        # poi_columns = [col for col in result_df.columns if col not in ["centroid_id", "accessibility_score"]]
        poi_columns = ['bank', 'supermarket', 'hospital', 'school']
        result_df["reachable_pois"] = result_df[poi_columns].apply(lambda row: row.to_dict(), axis=1)

        # Map results to GeoJSON
        score_map = dict(zip(result_df["centroid_id"], result_df["accessibility_score"]))
        reachable_pois_map = dict(zip(result_df["centroid_id"], result_df["reachable_pois"]))

        for feature in geojson_data["features"]:
            centroid_id = feature["properties"]["lad22cd"]
            feature["properties"]["neighborhood"] = feature["properties"]["lad22nm"]
            feature["properties"]["accessibility_score"] = score_map.get(centroid_id, 0)
            feature["properties"]["reachable_pois"] = reachable_pois_map.get(centroid_id, {})

        return jsonify(geojson_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=False)
