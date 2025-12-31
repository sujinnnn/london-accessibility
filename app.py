from backend import app as server

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import requests
import pandas as pd 

# Initialize Dash app
dash_app = dash.Dash(__name__, 
                server = server,
                external_stylesheets=["/assets/style.css"], 
                title="London Accessibility",
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Example POI categories
poi_categories = ['Bank', 'Supermarket', 'Hospital', 'School']

# 1) HERO SECTION
hero_section = html.Div(className="hero-section", children=[
    html.Div(className="hero-overlay"),
    html.H1("Public Transit Accessibility in London", className="hero-title")
])


# 2) INTRO SECTION

        
intro_section = html.Div(className="section", children=[
    html.H1(["Accessibility of London's Neighborhood",
            html.Br(),
             "by reachable number of Points of Interest"], 
        style={"textAlign": "center",  "marginBottom": "30px"}),
    html.Br(),
    html.Br(),
    html.H2("Intro"),
    html.P("""
    Access to public transit is an important element of transportation that enables
    individuals to reach essential services such as hospitals, schools, supermarkets,
    and financial institutions.
    """),
    html.P("""
    In this project, we quantify accessibility in London by combining travel-time
    and isochrone-based accessibility while reflecting user preferences for POIs.
    """)
])


# 3) PROBLEM SECTION
problem_section = html.Div(className="section", children=[
    html.H2("Out Goal"),
    html.H3("Find out how easy it is for people to get to important places"),
    html.P("""
    Firstly, we operationalize accessibility by combining two methods commonly
    used in accessibility assessment: travel-duration accessibility and
    isochrone-based access to POIs.
    """),
    html.P("""
    Secondly, we apply this definition across parameters such as different time
    thresholds, POI types, and user-assigned weights that reflect personal priorities.
    """)
])


# 4) HOW TO USE SECTION  (설명만 — 실제 슬라이더는 아래 dashboard에 있음)
how_to_section = html.Div(className="section howto", children=[
    html.H2("How to use it?"),

    html.Div(style = {
      "maxWidth": "850px",
      "margin" : "0 auto"
    }, children=[
        html.P("We can select time limit"),
    dcc.Slider(
            id="time-slider-test",
            min=5, max=60, step=5,
            marks={i: str(i) for i in range(5, 61, 5)},
            value=15
        ),
    html.Br(),
    html.P("We can assign preference weights to POIs"),
    html.Div([
            html.Div([
                html.Label(category),
                dcc.Input(
                    id=f"prev-weight-{category.lower()}",
                    type="number",
                    value=25,
                    style={"width": "80px"}
                )
            ], style={"display": "inline-block",
                      "marginRight": "30px"})
            for category in poi_categories
        ]),
    html.Br(),
    html.P(['Click ',
         html.Span("Calculate", className = "click"),
         ' button to update the map based on your settings.'])
])
])

# 5) OPTIONAL — DATA / METHOD SECTION
method_section = html.Div(className="section", children=[
    html.H3("How the Accessibility Score is Computed"),
    html.Ul(children=[
        html.Li("Dynamic time-based isochrone exploration"),
        html.Li("Walking + subway multimodal accessibility"),
        html.Li("User-defined POI weighting"),
        html.Li("Interactive neighborhood comparison"),
    ])
])


# 6) DASHBOARD SECTION (실제 UI + 지도)
dashboard_section = html.Div(className="section", children=[

    html.H2("Now you can try!"),

    # html.H1("Accessibility of London's Neighborhood",
    #         style={"textAlign": "center"}),

    # html.H3("by reachable number of Points of Interest",
    #         style={"textAlign": "center", "marginBottom": "30px"}),

    # Time slider
    html.Div([
        html.Label("Please Select Time Limit (minutes):"),
        dcc.Slider(
            id="time-slider",
            min=5, max=60, step=5,
            marks={i: str(i) for i in range(5, 61, 5)},
            value=15
        )
    ], style={"marginBottom": "30px"}),

    # POI Weights
    html.Div([
        html.Label("Please Give the Weight of Each POI Category:"),
        html.Div([
            html.Div([
                html.Label(category),
                dcc.Input(
                    id=f"weight-{category.lower()}",
                    type="number",
                    value=25,
                    style={"width": "80px"}
                )
            ], style={"display": "inline-block",
                      "marginRight": "30px"})
            for category in poi_categories
        ])
    ], style={"marginBottom": "30px"}),

    html.Button("Calculate!", id="update-map", n_clicks=0),

    dcc.Loading(children=[
        html.Div(style={"display": "flex", "height": "600px"}, children=[

            html.Div(style={"flex": 2}, children=[
                html.Div(id="map-info"),
                dcc.Graph(id="choropleth-map")
            ]),

            html.Div(style={"flex": 1}, children=[
                html.H3("Number of Reachable POIs"),
                dcc.Graph(id="bar-chart")
            ])
        ])
    ])
])



# ========== FINAL LAYOUT ==========
dash_app.layout = html.Div(style={"padding": "20px", "width": "90%"}, children=[
    hero_section,
    intro_section,
    problem_section,
    how_to_section,
    method_section,
    dashboard_section
])

@dash_app.callback(
    [Output("choropleth-map", "figure"),
     Output("map-info", "children")],
    Input("update-map", "n_clicks"),
    State("time-slider", "value"),
    [State(f"weight-{cat.lower()}", "value") for cat in poi_categories]
)
def update_map(n_clicks, time_limit, *weights):
    # Prepare POI weights
    poi_weights = {cat.lower(): weight for cat, weight in zip(poi_categories, weights)}

    # Send request to the backend
    response = requests.post(
        "/calculate_accessibility", 
        json={
        "time_limit": time_limit,
        "poi_weights": poi_weights
    })

    if response.status_code != 200:
        return px.choropleth_mapbox(
            geojson={},  # Empty GeoJSON
            locations=[],  # No locations
            title=f"Error: {response.json().get('error', 'Unknown error')}"
        )

    # Parse the GeoJSON response
    geojson_data = response.json()

    # Extract a DataFrame for the locations and accessibility scores
    data = []
    for feature in geojson_data["features"]:
        data.append({
            "centroid_id": feature["properties"]["lad22cd"],
            "accessibility_score": feature["properties"].get("accessibility_score", 0),
            "reachable_pois": feature["properties"].get("reachable_pois", {}),
            "neighborhood": feature["properties"].get("neighborhood", "")
        })
    global df
    df = pd.DataFrame(data)

    # Create the choropleth map
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_data,
        locations="centroid_id",
        featureidkey="properties.lad22cd",
        color="accessibility_score",
        color_continuous_scale="teal",
        title="Accessibility Score by Neighborhood",
        hover_data=["neighborhood", "centroid_id", "accessibility_score"]
    )
    fig.update_traces(
        marker_opacity=0.7 
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9,
        mapbox_center={"lat": 51.5074, "lon": -0.1278},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",   # ← 그래프 바깥
        plot_bgcolor="rgba(0,0,0,0)"
    )

    map_info = html.Div([
        html.P(f"Time Limit: {time_limit} minutes"),
        html.P("Weights: " + ", ".join([f"{cat}: {poi_weights[cat]}" for cat in poi_weights]))
    ])

    return fig, map_info


@dash_app.callback(
    Output("bar-chart", "figure"),
    Input("choropleth-map", "hoverData")
)
def update_bar_chart(hover_data):
    if hover_data is None:
        # Return an empty figure when there's no hover data
        return px.bar(title="Hover over a neighborhood to see data")

    # Extract the hovered neighborhood ID
    centroid_id = hover_data["points"][0]["location"]

    # Get the corresponding row from the DataFrame
    neighborhood_data = df[df["centroid_id"] == centroid_id]


    if neighborhood_data.empty:
        return px.bar(title="No data available")

    # Prepare data for the bar chart
    poi_counts = neighborhood_data.iloc[0]["reachable_pois"]

    # Create a DataFrame for POI counts
    poi_df = pd.DataFrame({
        "POI Category": list(poi_counts.keys()),
        "Count": list(poi_counts.values())
    })

    fig = px.bar(
        poi_df,
        y="POI Category",
        x="Count",
        orientation="h",
        title=f"Reachable POIs in Neighborhood {centroid_id}",
        labels={"Count": "Number of POIs", "POI Category": "Category"},
        text="Count"
    )
    fig.update_traces(
        textposition="outside",  
        textfont=dict(size=12, color="black") 
    )
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 30})
    return fig

if __name__ == "__main__":
    dash_app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )

