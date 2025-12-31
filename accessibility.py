import networkx as nx
import geojson
from pyproj import Transformer
from geopy.distance import geodesic
import osmnx as ox
from shapely.geometry import Point, Polygon
import geopandas as gpd
import json
import pandas as pd
import os
import requests

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/slee0215/london-accessibility-data/resolve/main"


def download_if_missing(filename):
    local_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(local_path):
        print(f"✔ Using local file: {filename}")
        return local_path

    url = f"{HF_BASE}/{filename}"
    print(f"⬇ Downloading from HuggingFace: {url}")

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✔ Downloaded and cached: {local_path}")
    return local_path
    
# Data
# ---- Graphs ----
walking_graph = nx.read_graphml(
    download_if_missing("modified_london_walking_graph.graphml")
)

subway_graph = nx.read_graphml(
    download_if_missing("london_rail_network.graphml")
)

# ---- GeoDataFrames ----
centroid_points = gpd.read_file(
    download_if_missing("neighborhood_centroid.geojson")
)

poi_gdf = gpd.read_file(
    download_if_missing("projected_osm_pois_point.geojson")
)

# ---- JSON Files ----
with open(download_if_missing("london_neighborhood.geojson"), "r") as f:
    geojson_data = json.load(f)

with open(download_if_missing("subway_walk_node_mapper.json"), "r") as f:
    subway_walk_node_mapper = json.load(f)

with open(download_if_missing("centroid_station_travel_time.json"), "r") as f:
    centroid_station_travel_time = json.load(f)

walk_subway_node_mapper = {value: key for key, value in subway_walk_node_mapper.items()}

# Utilities
def find_nearest_node(walking_graph, point, weight="travel_time"):
    """
    Find the nearest nodes fro start_point and end_point in walking graph
    """
    node = ox.distance.nearest_nodes(walking_graph, X=point[1], Y=point[0])

    return node

def generate_isochrone(graph, source_node, travel_time, weight="travel_time"):
    """
    Generate an isochrone polygon for a given graph node and max travel time.

    Args:
        graph (nx.Graph): Network graph (e.g., walking or subway).
        source_node (int): Node ID to calculate the isochrone from.
        max_travel_time (float): Maximum travel time (in seconds).
        weight (str): Edge weight to calculate travel time.

    Returns:
        gpd.GeoSeries: Isochrone as a GeoSeries object.
    """
    # Get subgraph of reachable nodes within the max travel time
    reachable_nodes = nx.single_source_dijkstra_path_length(graph, source=source_node, cutoff=travel_time, weight=weight)
    subgraph = graph.subgraph(reachable_nodes.keys())

    # Create isochrone polygon
    nodes = gpd.GeoDataFrame({
        'geometry': [Point(data['x'], data['y']) for node, data in subgraph.nodes(data=True)],
        'travel_time': list(reachable_nodes.values())
    })

    isochrone = nodes.unary_union.convex_hull 

    return isochrone

def get_station_within_time_limit(centroid_id, centroid_time_mapper, time_limit):

    return {key: value for key, value in centroid_time_mapper[centroid_id].items() if value <= time_limit}

def stations_within_time(remaining_time, start_node, graph):
    """
    Find all subway nodes that are reachable within remainting time
    """
    
    lengths, _ = nx.single_source_dijkstra(graph, source=start_node, cutoff=remaining_time, weight='travel_time')
    
    return lengths

def iterate_subway_stations(centroid_id, 
                            list_subway_stations, 
                            max_travel_time, 
                            poi_gdf, 
                            walk_graph, 
                            subway_graph, 
                            station_node_mapper,
                            walk_subway_node_mapper,
                            centroid_station_time
                           ):
    """
    Iterate through all reachable subway station from start point and get the number of reachable POIs
    """
    reachable_poi = []
    for subway_node in list_subway_stations:
        staion_node = walk_subway_node_mapper[subway_node]
        remaining_time = max_travel_time - centroid_station_time[centroid_id][subway_node]

        reachable_station = stations_within_time(remaining_time / 60.0, staion_node, subway_graph)
        for station, transit_time_min in reachable_station.items():
            nearest_walk_node_2 = station_node_mapper[station]
            remaintin_time_from_station = remaining_time - transit_time_min * 60.0
            isochrone_from_station = generate_isochrone(walk_graph, 
                                                        nearest_walk_node_2, 
                                                        remaintin_time_from_station,
                                                        weight="travel_time")

            pois_within_isochrone = poi_gdf[poi_gdf.within(isochrone_from_station)]
            reachable_poi += pois_within_isochrone.osm_id.tolist()
        
    return reachable_poi

def reachable_poi_by_walk(centroid_id,
                         max_travel_time,
                         poi_gdf,
                         walk_graph):
    """
    Get POIs reachable by walking
    """
    isochrone = generate_isochrone(walk_graph, 
                                    centroid_id, 
                                    max_travel_time,
                                    weight="travel_time")

    reachable_poi = poi_gdf[poi_gdf.within(isochrone)].osm_id.tolist()

    return reachable_poi
    

def calculate_accessibility(weight:dict,
                           df_count_poi:pd.DataFrame):
    # convert to percentage
    weight = {key: value / 100 for key, value in weight.items()}

    poi_list = [col for col in df_count_poi.columns if col != "centroid_id"]
    # max cnt poi
    df_count_poi_melt = df_count_poi.melt(id_vars=None, 
                                          value_vars=[col for col in df_count_poi.columns if col != "centroid_id"], 
                                          var_name="category", 
                                          value_name='value')
    
    count_poi_max = df_count_poi_melt.groupby(["category"])["value"].max().to_dict()

    df_result = df_count_poi.copy().fillna(0)
    df_result["accessibility_score"] = 0
    for poi_cat in poi_list:

        try:
            df_result["accessibility_score"] += df_result[poi_cat] * weight[poi_cat] / count_poi_max[poi_cat]
        except:
            continue

    return df_result
    
def run_acc(poi_gdf,
         centroid_points,
         walking_graph, 
         subway_graph, 
         subway_walk_node_mapper,
         walk_subway_node_mapper,
         centroid_station_travel_time,
         weight,
         time_limit=1800):

    # loop through every centroids
    df = pd.DataFrame()
    
    for _, row in centroid_points.iterrows():

        centroid_id = row["lad22cd"]

        centroid_node = row["walking_node"]

        list_subway_station = get_station_within_time_limit(centroid_id, 
                                                            centroid_station_travel_time, 
                                                            time_limit)

        reachable_poi = iterate_subway_stations(centroid_id, 
                                                list_subway_station, 
                                                time_limit, 
                                                poi_gdf, 
                                                walking_graph, 
                                                subway_graph, 
                                                subway_walk_node_mapper,
                                                walk_subway_node_mapper,
                                                centroid_station_travel_time
                                               )
        reachable_poi_walk = reachable_poi_by_walk(centroid_node,
                                                     time_limit,
                                                     poi_gdf,
                                                     walking_graph)
        
        reachable_poi += reachable_poi_walk
        reachable_poi = list(set(reachable_poi))

        df_reachable_poi = (pd.DataFrame({"poi_osm_id": reachable_poi})
                           .merge(poi_gdf[["osm_id", "fclass"]], left_on="poi_osm_id", right_on="osm_id"))

        df_reachable_poi["centroid_id"] = centroid_id

        df_reachable_poi_agg = df_reachable_poi.groupby(["centroid_id", "fclass"], as_index=False)["poi_osm_id"].nunique()

        df_reachable_poi_agg = df_reachable_poi_agg.pivot(columns="fclass", 
                                                          values="poi_osm_id", 
                                                          index="centroid_id").reset_index(drop=False)

        df = pd.concat([df_reachable_poi_agg, df])

    accessibility = calculate_accessibility(weight, df)

    return accessibility
