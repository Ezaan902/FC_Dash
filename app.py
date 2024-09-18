# %%

"""
Pydeck / ArcLayer
========

Documentation: https://deckgl.readthedocs.io/en/latest/index.html
flow source : CMANA Nouvelle-aquitaine.
dep, com source : Institut national de la statistique et des études économiques (INSEE)

"""

# 1. Importer les bibliothèques
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import pydeck as pdk
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import requests
import plotly.graph_objects as go
from unidecode import unidecode


# 2. Fonctions utilitaires et spécifications de couleurs

def list_files_in_github_folder(repo_owner, repo_name, folder_path):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"
    response = requests.get(url)
    if response.status_code == 200:
        files = response.json()
        return [file['name'] for file in files if file['name'].endswith('.geojson')]
    else:
        print(f"Erreur lors de la récupération des fichiers : {response.status_code}")
        return []

def load_geojson_from_github(repo_owner, repo_name, folder_path):
    geojson_data = []
    file_names = list_files_in_github_folder(repo_owner, repo_name, folder_path)
    
    for file_name in file_names:
        file_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{folder_path}/{file_name}"
        response = requests.get(file_url)
        if response.status_code == 200:
            data = response.json()
            geojson_data.append(data)
        else:
            print(f"Erreur lors du téléchargement du fichier {file_name} : {response.status_code}")
    
    return geojson_data

def filter_geojson_by_insee(geojson_files, insee_codes):
    filtered_features = []
    for geojson in geojson_files:
        for feature in geojson.get('features', []):
            if feature['properties'].get('code-insee') in insee_codes:
                filtered_features.append(feature)
    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }

def fc_filter(input_df):
    filtered_df_fc = input_df[input_df['FORMATION'].str.startswith('FC')]
    return filtered_df_fc

def remove_accents(input_df, columns):
    for column in columns:
        input_df[column] = input_df[column].apply(lambda x: unidecode(x) if isinstance(x, str) else x)
    return input_df

def processing_sankey_chart(input_df, flux_threshold):
    # Filtrer les données pour inclure uniquement les flux supérieurs au seuil
    input_df = input_df[(input_df['FLUX'] > flux_threshold)].copy()
    
    # Trier les cibles
    input_df.loc[:, 'DESTINATION_NAME'] = pd.Categorical(input_df['DESTINATION_NAME'], categories=sorted(set(input_df['DESTINATION_NAME']), key=lambda x: (x=='OUT', x)))
    input_df.sort_values("DESTINATION_NAME", inplace=True)

    columns = ['ORIGINE_NAME', 'DESTINATION_NAME']
    sankey_link_weight = 'FLUX'

    # Liste des valeurs de chaque colonne
    column_values = [input_df[col] for col in columns]

    # Générer les labels pour le Sankey en prenant toutes les valeurs uniques
    labels = sum([list(node_values.unique()) for node_values in column_values], [])

    # Initialiser un dictionnaire de dictionnaires (un par colonne)
    link_mappings = {col: {} for col in columns}

    # Chaque dictionnaire mappe un nœud à une valeur numérique unique
    i = 0
    for col, nodes in zip(columns, column_values):
        for node in nodes.unique():
            link_mappings[col][node] = i
            i = i + 1

    # Spécifier les colonnes sources et cibles
    source_nodes = column_values[: len(columns) - 1]
    target_nodes = column_values[1:]
    source_cols = columns[: len(columns) - 1]
    target_cols = columns[1:]
    links = []

    # Boucle pour créer une liste de liens au format [((src,tgt),wt),(),()...]
    for source, target, source_col, target_col in zip(source_nodes, target_nodes, source_cols, target_cols):
        for val1, val2, link_weight in zip(source, target, input_df[sankey_link_weight]):
            links.append(
                (
                    (
                        link_mappings[source_col][val1],
                        link_mappings[target_col][val2]
                    ),
                    link_weight,
                )
            )

    # Créer un DataFrame avec 2 colonnes : pour les liens (src, tgt) et les poids
    df_links = pd.DataFrame(links, columns=["link", "weight"])

    # Agréger les mêmes liens en un seul lien (par poids)
    df_links = df_links.groupby(by=["link"], as_index=False).agg({"weight": "sum"})

    # Générer trois listes nécessaires pour la visualisation Sankey
    sources = [val[0] for val in df_links["link"]]
    targets = [val[1] for val in df_links["link"]]
    weights = df_links["weight"]

    return labels, sources, targets, weights, columns

def make_sankey_chart(input_df, flux_threshold):
    # Traitement des données pour le diagramme Sankey
    labels, sources, targets, weights, columns = processing_sankey_chart(input_df, flux_threshold)
   
    # Créer des listes de couleurs pour les liens et les nœuds
    link_colors = ['rgba(15, 50, 80, 0.10)' for _ in sources]

    # Créer une liste de labels filtrés en fonction du seuil
    source_labels = input_df['ORIGINE_NAME'].unique()
    target_labels = input_df['DESTINATION_NAME'].unique()
    node_colors = ['rgba(15, 50, 80, 0.8)' if label in source_labels else 'rgba(238, 119, 110, 0.8)' if label in target_labels else '#EA4B3C' for label in labels]

    # Créer la figure
    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "white", width = 2),
        label = labels, 
        color = node_colors
        ),
        link = dict(
            source = sources,
            target = targets,
            value = weights,
            color = link_colors
    ))])

    fig.update_layout(
        autosize=False,
        width=800,  
        height=800,  
        title_text=None, 
        font_size=15, 
        font_family="Roboto Slab", 
        font_color="#222222", 
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig.show()

# Color specifications
RED_RGB = [234, 75, 60, 255]  # RGBA
BLUE_RGB = [15, 50, 80, 255]  # RGBA
LIGHTBLUE_RGB = [15, 50, 80, 20]  # RGBA
WHITE_RGB = [255, 255, 255, 255]  # RGBA
GREY_RGB = [85, 85, 85, 255]  # RGBA
ORANGE_RGB = [234, 75, 60, 255]  # RGBA
LIGHTORANGE_RGB = [238, 119, 110, 50]  # RGBA
HIGHLIGHT_RGB = [234, 75, 60, 255]  # RGBA
COLOR_RANGE = [
    [15, 50, 80],
    [176, 210, 217],
    [238, 119, 110],
    [234, 75, 50],
]



# 3. Charger les données
# Flow data
df = pd.read_csv('https://raw.githubusercontent.com/Ezaan902/FC_Dash/main/data/flow.csv', sep=';', encoding='utf-8')
df['FLUX'] = df['FLUX'].astype('int64')
df = fc_filter(df)
columns_to_clean = ['ORIGINE_NAME', 'DESTINATION_NAME', 'FORMATION']
df = remove_accents(df, columns_to_clean)

# Limites administratives pour les départements et les communes
file_path_dep = 'https://raw.githubusercontent.com/Ezaan902/FC_Dash/main/data/dep.json'
response = requests.get(file_path_dep)
dep = response.json()

file_path_com = 'https://raw.githubusercontent.com/Ezaan902/FC_Dash/main/data/com.json'
response = requests.get(file_path_com)
com = response.json()

repo_owner = 'Ezaan902'
repo_name = 'FC_Dash'
folder_path = 'data/iso'
iso_geojson_files = load_geojson_from_github(repo_owner, repo_name, folder_path)

# logo
file_path_img = 'https://raw.githubusercontent.com/Ezaan902/FC_Dash/main/assets/CMAregion-horizontal-rouge.png'


# 4. Initialiser l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div(
    style={
        'margin': '20px'  # Ajustez la marge selon vos besoins
    },
    children=[
        dbc.Row([
            dbc.Col([
                html.Img(src=file_path_img, className='logo', style={'width': '150px', 'height': 'auto'} )  # Utilisez la classe CSS pour le logo
            ], width='auto'),
            dbc.Col([
                html.H1(
                    "CARTOGRAPHIE DES APPRENANTS", 
                    className='title'  # Utilisez la classe CSS pour le titre
                )
            ], width='auto')
        ], align='center', style={'marginBottom': '30px'}),

        html.H3(
            "Représentation cartographique des flux d'apprenants (formation courte) à destination des sites de formation de la CMA Nouvelle-Aquitaine.",
            className='subtitle'  # Utilisez la classe CSS pour les sous-titres
        ),
        html.P(
            "Un apprenant est un individu inscrit à une formation de la CMA Nouvelle-Aquitaine pour la période 2023 - 2024 (année scolaire). Chaque apprenant est rattaché à une commune de domiciliation et à un site de formation. L'apprenant est considéré comme un flux entre sa commune de domiciliation et le site de formation de la CMA Nouvelle-Aquitaine. Les flux sont représentés par des arcs reliant les communes de domiciliation aux sites de formation. La largeur de l'arc est proportionnelle au nombre d'apprenants concernés par le flux.",
            className='text'  # Utilisez la classe CSS pour les paragraphes
        ),
        html.P(
            "Données issues de CMA, extraction Yparéo pour l'année 2023-2024.",
            className='italic'  # Utilisez la classe CSS pour les informations en italique
        ),

        # Ajoutez ici les autres composants de votre application
        dbc.Row([
            dbc.Col([
                html.Label(html.B('LOCALISATION DES APPRENANTS:'), className='custom-dropdown-label'),
                dcc.Dropdown(
                    id='origine-dropdown',
                    options=[{'label': 'TOUT SELECTIONNER', 'value': 'all'}] + [{'label': i, 'value': i} for i in df['ORIGINE_NAME'].unique()],
                    value='all',
                    className='custom-dropdown'
                ),
            ], width=4),
            dbc.Col([
                html.Label(html.B('LOCALISATION DES SITES DE FORMATION CMA:'),className='custom-dropdown-label'),
                dcc.Dropdown(
                    id='destination-dropdown',
                    options=[{'label': 'TOUT SELECTIONNER', 'value': 'all'}] + [{'label': i, 'value': i} for i in df['DESTINATION_NAME'].unique()],
                    value='all',
                    className='custom-dropdown'
                ),
            ], width=4),
            dbc.Col([
                html.Label(html.B('TYPE DE FORMATION:'),className='custom-dropdown-label'),
                dcc.Dropdown(
                    id='formation-dropdown',
                    options=[{'label': 'TOUT SELECTIONNER', 'value': 'all'}] + [{'label': i, 'value': i} for i in df['FORMATION'].unique()],
                    value='all',
                    className='custom-dropdown'
                ),
            ], width=4),
        ], className='mb-4'),  # Ajout d'une marge inférieure

        # Ajout du slider
        dbc.Row([
            dbc.Col([
                dcc.Slider(
                    id='flux-slider',
                    min=0,  # Valeur par défaut, sera mise à jour dans le callback
                    max=300,  # Valeur par défaut, sera mise à jour dans le callback
                    step=1,
                    value=0,  # Valeur initiale
                    marks={i: str(i) for i in range(0, 301, 10)},  # Marqueurs tous les 10
                    tooltip={"placement": "bottom", "always_visible": False},
                    className='custom-slider'
                ),
            ], width=12),
        ], className='mb-4'),  # Ajout d'une marge inférieure

        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    options=[
                        {"label": "Flux", "value": "show_arcs"},
                        {"label": "Communes", "value": "show_com"},
                        {"label": "departements", "value": "show_dep"},
                        {"label": "Isochrones (30 min)", "value": "show_iso"},
                    ],
                    value=["show_arcs", "show_dep"],  # Valeurs par défaut activées
                    id="layer-toggle",
                    inline=True,
                    className='custom-checkbox'
                ),
            ], width=12),
        ], className='mb-4'),

        # Ajout de la carte en haut
        dbc.Row([
            dbc.Col([
                html.Div(id='map-container', style={'height': '50vh', 'marginBottom': '5vh'})
            ], width=12, style={'height': '50vh', 'marginBottom': '5vh'}),
        ], align='center'),

        dbc.Row([
            dbc.Col([
                dbc.Row([
                    html.Div("APPRENANTS CONCERNES :", className='custom-dropdown-label',style={'textAlign': 'center', 'fontSize': '5'}),
                    html.Div(f"(Total d'apprenants {df['FLUX'].sum()})", style={'textAlign': 'center', 'fontSize': '10', 'marginBottom': '1vh', 'color': '#EA4B3C', 'fontStyle': 'italic'}),
                    dcc.Graph(id='pie-chart', style={'height': '25vh'}),
                ])
            ], width=4),
            dbc.Col([
                dash_table.DataTable(
                    id='flux-table',
                    columns=[
                        {'name': 'Origine', 'id': 'ORIGINE_NAME'},
                        {'name': 'Destination', 'id': 'DESTINATION_NAME'},
                        {'name': 'Flux', 'id': 'FLUX'}
                    ],
                    data=[],
                    sort_action='native',
                    style_table={'height': '40vh', 'overflowY': 'auto'},  # Ajustez la hauteur pour correspondre aux pie charts
                    style_cell={
                        'textAlign': 'left',
                        'font-family': 'Arial, sans-serif'  # Changer la police ici
                    },
                )
            ], width=8)
        ], align='center', style={'marginTop': '40vh', 'marginBottom': '5vh'}),
    ]
)

@app.callback(
    [Output('map-container', 'children'),
     Output('flux-table', 'data'),
     Output('flux-slider', 'min'),
     Output('flux-slider', 'max'),
     Output('flux-slider', 'marks'),
     Output('pie-chart', 'figure')],
    [Input('origine-dropdown', 'value'),
     Input('destination-dropdown', 'value'),
     Input('formation-dropdown', 'value'),
     Input('flux-slider', 'value'),
     Input("layer-toggle", "value")]
)

# 5. Créer la fonction de rappel
def update_dash(origine, destination, formation, flux_value, layer_toggle):
    # Initialiser filtered_df avec le DataFrame df
    filtered_df = df.copy()

    arc_layer=None
    com_layer=None
    dep_layer=None
    iso_layer=None

    if origine != 'all':
        filtered_df = filtered_df[filtered_df['ORIGINE_NAME'] == origine]
    if destination != 'all':
        filtered_df = filtered_df[filtered_df['DESTINATION_NAME'] == destination]
    if formation != 'all':
        filtered_df = filtered_df[filtered_df['FORMATION'] == formation]
        # Après avoir appliqué les filtres
    if filtered_df.empty:
        empty_pie_chart = go.Figure(data=[go.Pie(
            labels=['Part représentée', 'Autres'],
            values=[0, 100],
            hole=.7,
            marker=dict(colors=['#969696', '#E0E0E0'])
        )])
        empty_pie_chart.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            annotations=[dict(text="0%", x=0.5, y=0.5, font_size=20, font=dict(weight='bold'), font_color='#222222', showarrow=False)]
        )

        return (
            html.Iframe(
                srcDoc=pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=44.8392,
                        longitude=-0.5812,
                        zoom=6,
                        min_zoom=6,
                    ),
                    map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
                ).to_html(as_string=True),
                width='100%',
                height='600'
            ),
            [],  # Données de la table vides
            0,  # Valeur minimale du slider
            0,  # Valeur maximale du slider
            {},  # Marques du slider vides
            empty_pie_chart  # Graphique neutre pour le pie chart
        )

    # Filtrer les départements en fonction des DEP_CODE présents dans les données filtrées
    filtered_df['DEP_CODE'] = filtered_df['DEP_CODE'].astype(str)
    filtered_df['ORIGINE_CODE'] = filtered_df['ORIGINE_CODE'].astype(str)

    dep_codes_in_filtered_df = filtered_df['DEP_CODE'].unique()
    filtered_dep = {
        "type": "FeatureCollection",
        "features": [
            feature for feature in dep.get('features', [])
            if feature.get('properties', {}).get('code_insee') in dep_codes_in_filtered_df
        ]
    }

    # Filtrer les communes en fonction des ORIGINE_CODE présents dans les données filtrées
    com_codes_in_filtered_df = filtered_df['ORIGINE_CODE'].unique()
    filtered_com = {
        "type": "FeatureCollection",
        "features": [
            feature for feature in com.get('features', [])
            if feature.get('properties', {}).get('insee') in com_codes_in_filtered_df
        ]
    }
    
    # Calculer les flux totaux pour les données filtrées et non filtrées
    total_filtered_flux = filtered_df['FLUX'].sum()
    total_flux = df['FLUX'].sum()

    # Grouper les données filtrées par ORIGINE_CODE, ORIGINE_NAME, DESTINATION_CODE, DESTINATION_NAME
    grouped_df = filtered_df.groupby(['ORIGINE_CODE', 'ORIGINE_NAME', 'DESTINATION_CODE', 'DESTINATION_NAME']).agg({
        'FLUX': 'sum',
        'X_START': 'first',
        'Y_START': 'first',
        'X_END': 'first',
        'Y_END': 'first'
    }).reset_index()

    grouped_df = grouped_df[grouped_df['FLUX'] >= flux_value]
    

    # Trier les communes par flux et calculer la part de chaque commune
    # Ajouter la valeur minimum et maximum pour définir les classes correctement
    sorted_df = grouped_df.sort_values(by='FLUX', ascending=False)
    #sorted_df['VALUE%'] = sorted_df['FLUX'] / total_filtered_flux * 100
    sorted_df['LOG_VALUE%'] = np.log1p(sorted_df['FLUX'])
    
    
    # Définir la rampe de couleur
    #cmap = plt.get_cmap('viridis')
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#0F3250", "#EA4B3C"])
    # Normaliser les valeurs de flux pour qu'elles soient entre 0 et 1
    norm = mcolors.Normalize(vmin=sorted_df['LOG_VALUE%'].min(), vmax=sorted_df['LOG_VALUE%'].max())
    def color_scale(val):
        rgba = cmap(norm(val))
        return [int(255 * c) for c in rgba[:3]] + [127]  # Convertir en RGBA

    # Appliquer la fonction de couleur sur les données
    # Créer un dictionnaire de mappage pour les couleurs
    # Mettre à jour les couleurs des communes dans filtered_com
    sorted_df['color'] = sorted_df['LOG_VALUE%'].apply(color_scale)
    color_mapping = dict(zip(sorted_df['ORIGINE_CODE'], sorted_df['color']))
    for feature in filtered_com['features']:
        insee_code = feature['properties']['insee']
        feature['properties']['color'] = color_mapping.get(insee_code, [200, 200, 200, 100])

    # Calculer les KPI
    # Calculer le nombre total d'apprenants concernés par les données filtrées
    # Calculer la part des apprenants concernés par rapport au total
    selected_flux = sorted_df['FLUX'].sum()
    percentage = (selected_flux / total_flux) * 100 if total_flux > 0 else 0
    kpi_text = str(selected_flux)

    pie_chart = go.Figure(data=[go.Pie(
        labels=['Part représentée', 'Autres'],
        values=[percentage, 100 - percentage],
        hole=.7,
        marker=dict(colors=['#0F3250', '#EA4B3C'])
    )])
    pie_chart.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        annotations=[dict(text=kpi_text, x=0.5, y=0.5, font_size=20, font=dict(weight='bold'), font_color='#222222', showarrow=False)]
    )

    # Plotter les différentes couches
    # ArcLayer pour les flux, GeoJsonLayer pour les départements et les communes
    # Ajouter des informations de survol pour les arcs
    # Créer l'objet Deck
    layers = []

    if "show_arcs" in layer_toggle:
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=grouped_df,
            get_width="FLUX/7.5",
            get_source_position=["X_START", "Y_START"],
            get_target_position=["X_END", "Y_END"],
            get_tilt=20,
            get_source_color=LIGHTORANGE_RGB,
            get_target_color=BLUE_RGB,
            pickable=True,
            auto_highlight=True,
            highlight_color=HIGHLIGHT_RGB
        )
        layers.append(arc_layer)

    if "show_dep" in layer_toggle:
        dep_layer = pdk.Layer(
            "GeoJsonLayer",
            data=filtered_dep,
            pickable=False,
            stroked=True,
            filled= False,
            extruded=False,
            get_line_color=GREY_RGB,
            get_line_width=200
        )
        layers.append(dep_layer)

    if "show_com" in layer_toggle:
        com_layer = pdk.Layer(
            "GeoJsonLayer",
            data=filtered_com,
            pickable=False,
            stroked=True,
            filled=True,
            extruded=False,
            get_fill_color='properties.color',
            get_line_color=WHITE_RGB,
            get_line_width=100
        )
        layers.append(com_layer)

    if "show_iso" in layer_toggle:
        selected_insee_codes = filtered_df['DESTINATION_CODE'].unique().astype(str)
        filtered_iso_geojson = filter_geojson_by_insee(iso_geojson_files, selected_insee_codes)
        
        iso_layer = pdk.Layer(
            "GeoJsonLayer",
            data=filtered_iso_geojson,
            pickable=False,
            stroked=True,
            filled=True,
            extruded=False,
            get_fill_color=LIGHTORANGE_RGB,
            get_line_color=ORANGE_RGB,
            get_line_width=100
        )
        layers.append(iso_layer)

    view_state = pdk.ViewState(
        latitude=44.8392,
        longitude=-0.5812,
        bearing=0,
        pitch=60,
        zoom=6,
        min_zoom=6,
    )

    TOOLTIP_TEXT = {
        "html": "<b style='color: white;'>{FLUX}</b> Apprenant(s) à destination de <b style='color: white;'>{DESTINATION_NAME} </b><br>"
                "Commune de domicilisation <b style='color: #EE776E;'>{ORIGINE_NAME} </b><br>"
                "<i style='font-size: 0.7em;'>Source: CMA Nouvelle-Aquitaine 2023-2024</i>"
    }

    r = pdk.Deck(
        layers=[arc_layer, com_layer, dep_layer, iso_layer],
        initial_view_state=view_state,
        map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        tooltip=TOOLTIP_TEXT
    )

    # Créer un tableau de données pour le composant DataTable
    # Mettre à jour la table pour inclure la colonne de couleur
    # Définir des règles de style conditionnel pour colorer les cellules
    # Mettre à jour le composant DataTable
    table_data = sorted_df[['ORIGINE_NAME', 'DESTINATION_NAME', 'FLUX']].sort_values(by='FLUX', ascending=False).to_dict('records')

    # Définir les valeurs min et max du slider
    slider_min = 0
    # Arrondir slider_max à la dizaine supérieure
    slider_max = int(sorted_df['FLUX'].max())

    # Créer des marques arrondies à la dizaine la plus proche
    slider_marks = {i: {'label':f'{i}'} for i in range(slider_min, slider_max + 1, 10)}

    return (
        html.Iframe(
            srcDoc=r.to_html(as_string=True),
            width='100%',
            height='600'
        ),
        table_data,
        slider_min,
        slider_max,
        slider_marks,
        pie_chart,
    )

if __name__ == '__main__':
    app.run_server(debug=True)