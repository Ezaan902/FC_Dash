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


# 1.  Fonctions utilitaires et spécifications de couleurs
def remove_accents(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: unidecode(x) if isinstance(x, str) else x)
    return df

def prepare_sankey_data(df):
    labels = list(pd.concat([df['ORIGINE_NAME'], df['DESTINATION_NAME']]).unique())
    source_indices = [labels.index(origin) for origin in df['ORIGINE_NAME']]
    target_indices = [labels.index(destination) for destination in df['DESTINATION_NAME']]
    values = df['FLUX'].tolist()
    
    return labels, source_indices, target_indices, values

def create_sankey_diagram(df):
    labels, sources, targets, values = prepare_sankey_data(df)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=2),
            label=labels,
            color="rgba(15, 50, 80, 0.8)"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(15, 50, 80, 0.10)',
            label=df['FORMATION']
        )
    )])
    
    fig.update_layout(
        title_text="Diagramme de Sankey des Flux de Formation",
        font=dict(
            size=15,
            family="Roboto Slab",
            color="#222222"
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

# Color specifications
RED_RGB = [234, 75, 60, 255]  # RGBA
BLUE_RGB = [15, 50, 80, 255]  # RGBA
LIGHTBLUE_RGB = [15, 50, 80, 20]  # RGBA
GREY_RGB = [255, 255, 255, 255]  # RGBA
ORANGE_RGB = [234, 75, 60, 255]  # RGBA
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
columns_to_clean = ['ORIGINE_NAME', 'DESTINATION_NAME', 'FORMATION']
df = remove_accents(df, columns_to_clean)

# Limites administratives pour les départements et les communes
file_path_dep = 'https://raw.githubusercontent.com/Ezaan902/FC_Dash/main/data/dep.json'
response = requests.get(file_path_dep)
dep = response.json()

file_path_com = 'https://raw.githubusercontent.com/Ezaan902/FC_Dash/main/data/com.json'
response = requests.get(file_path_com)
com = response.json()

"""
file_path_com = 'https://raw.githubusercontent.com/Ezaan902/FC_Dash/main/data/com.json'
response = requests.get(file_path_com)
com = response.json()
"""

file_path_img = 'https://raw.githubusercontent.com/Ezaan902/FC_Dash/main/assets/CMAregion-horizontal-rouge.png'


# 4. Initialiser l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# server = app.server

app.layout = html.Div(
    style={
        'margin': '20px'  # Ajustez la marge selon vos besoins
    },
    children=[
        dbc.Row([
            dbc.Col([
                html.Img(src=file_path_img, className='logo')  # Utilisez la classe CSS pour le logo
            ], width='auto'),
            dbc.Col([
                html.H1(
                    "CARTOGRAPHIE DES APPRENANTS", 
                    className='title'  # Utilisez la classe CSS pour le titre
                )
            ], width='auto')
        ], align='center'),

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
                html.Label(html.B('LOCALISATION DES APPRENANTS:')),
                dcc.Dropdown(
                    id='origine-dropdown',
                    options=[{'label': 'TOUT SELECTIONNER', 'value': 'all'}] + [{'label': i, 'value': i} for i in df['ORIGINE_NAME'].unique()],
                    value='all'
                ),
            ], width=4),
            dbc.Col([
                html.Label(html.B('LOCALISATION DES SITES DE FORMATION CMA:')),
                dcc.Dropdown(
                    id='destination-dropdown',
                    options=[{'label': 'TOUT SELECTIONNER', 'value': 'all'}] + [{'label': i, 'value': i} for i in df['DESTINATION_NAME'].unique()],
                    value='all'
                ),
            ], width=4),
            dbc.Col([
                html.Label(html.B('TYPE DE FORMATION:')),
                dcc.Dropdown(
                    id='formation-dropdown',
                    options=[{'label': 'TOUT SELECTIONNER', 'value': 'all'}] + [{'label': i, 'value': i} for i in df['FORMATION'].unique()],
                    value='all'
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
                        {"label": "Isochrones", "value": "show_iso"},
                    ],
                    value=["show_arcs"],  # Valeurs par défaut activées
                    id="layer-toggle",
                    inline=True,
                ),
            ], width=12),
        ], className='mb-4'),

        # Ajout de la légende des arcs et de l'échelle de couleur
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("Source", style={'backgroundColor': 'rgba(15, 50, 80, 255)', 'padding': '5px', 'marginRight': '10px', 'borderRadius': '5px'}),
                    html.Span("Cible", style={'backgroundColor': 'rgba(234, 75, 60, 255)', 'padding': '5px', 'borderRadius': '5px'})
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.Span("Échelle de couleur pour les communes", style={'marginRight': '10px'}),
                    html.Div(id='color-scale', style={'display': 'flex'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'})
            ], width=12),
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='map-container')
            ], width=8),
            dbc.Col([
                html.Div(id='kpi-widget', style={'fontSize': '24px', 'fontWeight': 'bold', 'marginBottom': '20px'}),  # Widget KPI
                dash_table.DataTable(
                    id='flux-table',
                    columns=[
                        {'name': 'Origine', 'id': 'ORIGINE_NAME'},
                        {'name': 'Destination', 'id': 'DESTINATION_NAME'},
                        {'name': 'Flux', 'id': 'FLUX'}
                    ],
                    data=[],
                    sort_action='native',
                    style_table={'height': '500px', 'overflowY': 'auto'},  # Hauteur fixe et défilement vertical
                    style_cell={
                        'textAlign': 'left',
                        'font-family': 'Arial, sans-serif'  # Changer la police ici
                    }
                )
            ], width=4),
        ]),
        dcc.Graph(id='sankey-diagram'),
    
        dbc.Row([
            dbc.Col([
                html.Button(
                    "Excel export", 
                    id="export-button", 
                    n_clicks=0,
                ),
                dcc.Download(id="download-dataframe-csv"),
            ], width=12),
        ], className='mb-4'),  # Ajout d'une marge inférieure
    ]
)

@app.callback(
    [Output('map-container', 'children'),
     Output('flux-table', 'data'),
     Output('kpi-widget', 'children'),
     Output('flux-slider', 'min'),
     Output('flux-slider', 'max'),
     Output('flux-slider', 'marks'),
     Output("download-dataframe-csv", "data"),
     Output('sankey-diagram', 'figure')],
    [Input('origine-dropdown', 'value'),
     Input('destination-dropdown', 'value'),
     Input('formation-dropdown', 'value'),
     Input('flux-slider', 'value'),
     Input("export-button", "n_clicks"),
     Input("layer-toggle", "value")],
    [State('origine-dropdown', 'value'),
     State('destination-dropdown', 'value'),
     State('formation-dropdown', 'value'),
     State('flux-slider', 'value')],
    prevent_initial_call=True,
)

# 5. Créer la fonction de rappel
def update_dash(origine, destination, formation, flux_value, n_clicks, layer_toggle, origine_state, destination_state, formation_state, flux_value_state):
    # Initialiser filtered_df avec le DataFrame df
    filtered_df = df.copy()

    arc_layer=None
    com_layer=None
    iso_layer=None

    if origine != 'all':
        filtered_df = filtered_df[filtered_df['ORIGINE_NAME'] == origine]
    if destination != 'all':
        filtered_df = filtered_df[filtered_df['DESTINATION_NAME'] == destination]
    if formation != 'all':
        filtered_df = filtered_df[filtered_df['FORMATION'] == formation]

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
    kpi_text = f"APPRENANTS CONCERNES: {selected_flux}"
    percentage_text = f"PART REPRESENTEE: {percentage:.1f}%"

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
            get_source_color=LIGHTBLUE_RGB,
            get_target_color=BLUE_RGB,
            pickable=True,
            auto_highlight=True,
            highlight_color=HIGHLIGHT_RGB
        )
        layers.append(arc_layer)

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

    if "show_com" in layer_toggle:
        com_layer = pdk.Layer(
            "GeoJsonLayer",
            data=filtered_com,
            pickable=False,
            stroked=True,
            filled=True,
            extruded=False,
            get_fill_color='properties.color',
            get_line_color=GREY_RGB,
            get_line_width=100
        )
        layers.append(com_layer)

    """
    if "show_iso" in layer_toggle:
        iso_layer = pdk.Layer(
            "GeoJsonLayer",
            data=iso,
            pickable=False,
            stroked=True,
            filled=True,
            extruded=False,
            get_fill_color='properties.color',
            get_line_color=ORANGE_RGB,
            get_line_width=100
        )
        layers.append(com_layer)
    """

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
                "Commune de domicilisation <b style='color: white;'>{ORIGINE_NAME} </b><br>"
                "<i style='font-size: 0.7em;'>Source: CMA Nouvelle-Aquitaine 2023-2024</i>"
    }

    r = pdk.Deck(
        layers=[arc_layer, dep_layer, com_layer],
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

    if n_clicks > 0:
        return (
            html.Iframe(
                srcDoc=r.to_html(as_string=True),
                width='100%',
                height='600'
            ),
            table_data,
            html.Div([
                html.Div(kpi_text, style={'fontSize': '16px'}),
                html.Div(percentage_text, style={'fontSize': '16px'})
            ]),
            slider_min,
            slider_max,
            slider_marks,
            dcc.send_data_frame(filtered_df.to_excel, "filtered_data.xlsx", index=False)
        )
    
    # Créer le diagramme de Sankey
    sankey_figure = create_sankey_diagram(filtered_df)

    return (
        html.Iframe(
            srcDoc=r.to_html(as_string=True),
            width='100%',
            height='600'
        ),
        table_data,
        html.Div([html.Div(kpi_text, style={'fontSize': '16px'}), html.Div(percentage_text, style={'fontSize': '16px'})]),
        slider_min,
        slider_max,
        slider_marks,
        sankey_figure,
        dash.no_update,
    )

if __name__ == '__main__':
    app.run_server(debug=True)