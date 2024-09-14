# %%
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


"""
Pydeck / ArcLayer
========

Documentation: https://deckgl.readthedocs.io/en/latest/index.html
Update data source : https://www.insee.fr/fr/statistiques/7630376 (2020)

"""

# Charger les données
df = pd.read_csv(r'c:\Users\axel.benoit\Documents\Demandes ponctuelles\Aurélie\flow.csv', sep=';', encoding='utf-8')
df['FLUX'] = df['FLUX'].astype('int64')

# Chemin vers le fichier JSON
file_path_dep = r'c:\Users\axel.benoit\Documents\Demandes ponctuelles\Aurélie\dep (1).json'
with open(file_path_dep, 'r') as file:
    dep = json.load(file)

# Chemin vers le fichier JSON
file_path_com = r'c:\Users\axel.benoit\Documents\Demandes ponctuelles\Aurélie\com (2).json'
with open(file_path_com, 'r') as file:
    com = json.load(file)

# Codes de couleur
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

# Initialiser l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    style={
        'margin': '20px'  # Ajustez la marge selon vos besoins
    },
    children=[
        html.H1(
            "CARTOGRAPHIE DES APPRENANTS", 
            style={
                'textAlign': 'left', 
                'marginBottom': '20px', 
                'font': 'RobotoSlab', 
                'fontWeight': 'bold', 
                'fontSize': '32px'  # Ajustez la taille de la police ici
            }
        ),
                html.H3(
            [
                "Représentation  cartographique des flux d'apprenants (formation courte) à destination des sites de formation de la CMA Nouvelle-Aquitaine.",
            ],
            style={
                'textAlign': 'left', 
                'marginBottom': '10px', 
                'font': 'RobotoSlab', 
                'fontWeight': 'semibold', 
                'fontSize': '16px'  # Ajustez la taille de la police ici
            }
        ),
        html.H3(
            [
                "Un apprenant est un individu inscrit à une formation de la CMA Nouvelle-Aquitaine pour la période 2O23 - 2024 (année scolaire). Chaque apprenant est rattaché à une commune de domiciliation et à un site de formation. L'apprenant est considéré comme un flux entre sa commune de domiciliation et le site de formation de la CMA Nouvelle-Aquitaine.",
                "Les flux sont représentés par des arcs reliant les communes de domiciliation aux sites de formation. La largeur de l'arc est proportionnelle au nombre d'apprenants concernés par le flux.",
            ],  
            style={
                'textAlign': 'left', 
                'marginBottom': '10px', 
                'font': 'RobotoSlab', 
                'fontWeight': 'normal', 
                'fontSize': '14px'  # Ajustez la taille de la police ici
            }
        ),
        html.H3(
            [
                html.I("Données issues de CMA, extraction Yparéo pour l'année 2023-2024.")
            ],
            style={
                'textAlign': 'left',
                'marginBottom': '10px',
                'font': 'RobotoSlab',
                'fontWeight': 'normal',
                'fontSize': '10px'  # Ajustez la taille de la police ici
            }
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
     Output("download-dataframe-csv", "data")],
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

def update_map_and_table(origine, destination, formation, flux_value, n_clicks, layer_toggle, origine_state, destination_state, formation_state, flux_value_state):
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

    # Calculer les KPI
    # Calculer le nombre total d'apprenants concernés par les données filtrées
    # Calculer la part des apprenants concernés par rapport au total
    selected_flux = sorted_df['FLUX'].sum()
    percentage = (selected_flux / total_flux) * 100 if total_flux > 0 else 0
    kpi_text = f"APPRENANTS CONCERNES: {selected_flux}"
    percentage_text = f"PART REPRESENTEE: {percentage:.1f}%"

    # Fonction pour mapper les couleurs en fonction des breaks
    # Appliquer la fonction de couleur sur les données
    # Créer un dictionnaire de mappage pour les couleurs
    # Mettre à jour les couleurs des communes dans filtered_com
    def color_scale(val):
        rgba = cmap(norm(val))
        return [int(255 * c) for c in rgba[:3]] + [127]  # Convertir en RGBA
    
    sorted_df['color'] = sorted_df['LOG_VALUE%'].apply(color_scale)
    color_mapping = dict(zip(sorted_df['ORIGINE_CODE'], sorted_df['color']))
    for feature in filtered_com['features']:
        insee_code = feature['properties']['insee']
        feature['properties']['color'] = color_mapping.get(insee_code, [200, 200, 200, 100])

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

    return html.Iframe(
        srcDoc=r.to_html(as_string=True),
        width='100%',
        height='600'
    ), table_data, html.Div([
        html.Div(kpi_text, style={'fontSize': '16px'}),
        html.Div(percentage_text, style={'fontSize': '16px'})  # Diminuer la taille de la police ici
    ]), slider_min, slider_max, slider_marks,dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)

# %%
import plotly.graph_objects as go
import plotly.io as pio
import kaleido

# Transform data for flux 
def categorise_flux(input_flux, input_com_code) :
    # Create a new column to categorise the flux
    input_flux['in/out_start'] = 'OUT'
    input_flux['in/out_end'] = 'OUT'
    input_flux.loc[input_flux['CODGEO'].isin(input_com_code), 'in/out_start'] = 'IN'
    input_flux.loc[input_flux['DCLT'].isin(input_com_code), 'in/out_end'] = 'IN'

    # Group by 'source', 'target', 'in/out_start', 'in/out_end' and sum 'weight'
    input_flux.loc[input_flux['in/out_end'] == 'OUT', 'L_DCLT'] = 'OUT'
    flux_grouped = input_flux.groupby(['CODGEO', 'LIBGEO', 'L_DCLT', 'in/out_start', 'in/out_end'], as_index=False)['NBFLUX_C20_ACTOCC15P'].sum()

    # Create a list of the communes where the start of the flux is going in and out from communes in the area of interest
    in_start_list = input_flux[input_flux['in/out_start'] == 'IN']['LIBGEO'].unique().tolist()
    out_start_list = input_flux[input_flux['in/out_start'] == 'OUT']['LIBGEO'].unique().tolist()

    # Create a list of the communes where the end of the flux is going in and out from communes in the area of interest
    in_end_list = input_flux[input_flux['in/out_end'] == 'IN']['L_DCLT'].unique().tolist()
    out_end_list = input_flux[input_flux['in/out_end'] == 'OUT']['L_DCLT'].unique().tolist()

    flux_grouped = flux_grouped.rename(columns={'CODGEO':'code_source','LIBGEO': 'source', 'L_DCLT':'target', 'NBFLUX_C20_ACTOCC15P': 'weight'})
    flux_grouped = flux_grouped[['code_source','source', 'target', 'weight', 'in/out_start', 'in/out_end']]
    flux_grouped['weight'] = flux_grouped['weight'].astype(int)

    return flux_grouped, in_start_list, out_start_list, in_end_list, out_end_list

# produce start and end view for flux
def start_end_view_flux(input_flux, param):
    
    if param == 'start' :
        # Select only the rows where the source is in the list
        input_flux = input_flux[(input_flux['in/out_start'] == 'IN')]

        return input_flux

    elif param == 'end' :
        # Select only the rows where the target is out the list
        input_flux = input_flux[(input_flux['in/out_start'] == 'OUT') & (input_flux['in/out_end'] == 'IN')]
        # Ordering the rows by the weight
        input_flux = input_flux.nlargest(20, 'weight')

        return input_flux
    
    else:
        print("The parameter is not correct. Please choose 'in' or 'out'.")
        
        return None

# Get the sum of the weights for each pair of 'in/out_start' and 'in/out_end'
def get_flux_weight(input_flux, param):
    # Calculate the sum of the weights for each pair of 'in/out_start' and 'in/out_end'
    summed_weights = input_flux.groupby(['in/out_start', 'in/out_end'])['weight'].sum().reset_index()

    if param == 'start' :

        # Filter the rows where the source and the target are both in the area of interest
        summed_weights = summed_weights[((summed_weights['in/out_start'] == 'IN') & (summed_weights['in/out_end'] == 'IN')) | 
                                        ((summed_weights['in/out_start'] == 'IN') & (summed_weights['in/out_end'] == 'OUT'))]
        return summed_weights

    elif param == 'end':

        # Filter the rows where the target is in the area of interest and the source is out
        summed_weights = summed_weights[((summed_weights['in/out_start'] == 'OUT') & (summed_weights['in/out_end'] == 'IN'))]
        
        return summed_weights

    else:
        print("The parameter is not correct. Please choose 'in' or 'out'.")
        
        return None

# way to plot flux...
def processing_sankey_chart(input_df, param):
    # filter the data to have the communes in the area of interest as sources
    #input_df['target'] = pd.Categorical(input_df['target'], categories=sorted(set(input_df['target']), key=lambda x: (x=='OUT', x)))
    #input_df.sort_values("target", inplace=True)
    input_df = input_df.copy()
    input_df.loc[:, 'target'] = pd.Categorical(input_df['target'], categories=sorted(set(input_df['target']), key=lambda x: (x=='OUT', x)))
    input_df.sort_values("target", inplace=True)

    if param == 'start':
        columns = ['source', 'target', 'in/out_end']
    elif param == 'end':
        columns = ['source', 'target']
    else:
        print("The parameter is not correct. Please put'flux_start' or 'flux_end' df as parameter.")

    sankey_link_weight = 'weight'


    # list of list: each list are the set of nodes in each tier/column
    column_values = [input_df[col] for col in columns]

    # this generates the labels for the sankey by taking all the unique values
    labels = sum([list(node_values.unique()) for node_values in column_values],[])

    # initializes a dict of dicts (one dict per tier) 
    link_mappings = {col: {} for col in columns}

    # each dict maps a node to unique number value (same node in different tiers
    # will have different nubmer values
    i = 0
    for col, nodes in zip(columns, column_values):
        for node in nodes.unique():
            link_mappings[col][node] = i
            i = i + 1

    # specifying which coluns are serving as sources and which as sources
    # ie: given 3 df columns (col1 is a source to col2, col2 is target to col1 and 
    # a source to col 3 and col3 is a target to col2
    source_nodes = column_values[: len(columns) - 1]
    target_nodes = column_values[1:]
    source_cols = columns[: len(columns) - 1]
    target_cols = columns[1:]
    links = []

    # loop to create a list of links in the format [((src,tgt),wt),(),()...]
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

    # creating a dataframe with 2 columns: for the links (src, tgt) and weights
    df_links = pd.DataFrame(links, columns=["link", "weight"])

    # aggregating the same links into a single link (by weight)
    #df_links = df_links.groupby(by=["link"], as_index=False).agg({"weight": sum})
    df_links = df_links.groupby(by=["link"], as_index=False).agg({"weight": "sum"})

    # generating three lists needed for the sankey visual
    sources = [val[0] for val in df_links["link"]]
    targets = [val[1] for val in df_links["link"]]
    weights = df_links["weight"]

    return labels, sources, targets, weights, columns

def labels_modifier(input_df, columns):
    # Modify the labels to include the weights
    #node_weights = [input_df.groupby(col)['weight'].sum().to_dict() for col in columns]
    node_weights = [input_df.groupby(col, observed=True)['weight'].sum().to_dict() for col in columns]

    # Modify the labels to include the weights
    labels_with_weights = [[f'<b>{label}</b> ({node_weight.get(label, 0)})' for label in input_df[col].unique()] for col, node_weight in zip(columns, node_weights)]

    return labels_with_weights

def make_sankey_chart(input_df, input_implant_com_code, param, name):
    # Data processing for the sankey chart
    labels, sources, targets, weights, columns = processing_sankey_chart(input_df, param)

    # Modify the labels to include the weights
    labels_with_weights = labels_modifier(input_df, columns)

    # Combine the labels
    labels_with_weights = sum(labels_with_weights, [])
   
    # Get the index of the implant commune
    # create dict of labels to index
    code_name_dict = input_df.set_index('code_source')['source'].to_dict()
    input_implant_com_name = code_name_dict.get(input_implant_com_code)
    if input_implant_com_name in labels:
        implant_com_index = labels.index(input_implant_com_name)
    else:
        implant_com_index = None
    
    # Create color lists for the links and the nodes
    link_colors = ['#EA4B3C' if source == implant_com_index else 'rgba(15, 50, 80, 0.10)' for source in sources]
    if param == 'start':
        node_colors = ['rgba(15, 50, 80, 0.8)' if label == 'OUT' else '#EA4B3C' if implant_com_index is not None and label == labels[implant_com_index] else 'rgba(238, 119, 110, 0.8)' for label in labels]
    else:
        source_labels = input_df['source'].unique()
        target_labels = input_df['target'].unique()
        node_colors = ['rgba(15, 50, 80, 0.8)' if label in source_labels else 'rgba(238, 119, 110, 0.8)' if label in target_labels else '#EA4B3C' for label in labels]
   

    # Create the figure
    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "white", width = 2),
        label = labels_with_weights, 
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

# %%
    # Transform flux data and make specific lists
flux, in_start_list, out_start_list, in_end_list, out_end_list = categorise_flux(flux_df, com_code)

    # Get the weight of the flux for the start and the end
flux_start = start_end_view_flux(flux, 'start')
flux_end = start_end_view_flux(flux, 'end')

    # Get the summed weights for the start and the end
flux_start_weight = get_flux_weight(flux_start, 'start')
flux_end_weight = get_flux_weight(flux_end, 'end')

    # Create the sankey plots for each view
make_sankey_chart(flux_start, implant_com_code, 'start', 'sankey_in')
make_sankey_chart(flux_end, implant_com_code, 'end', 'sankey_out')

# %%
import pydeck as pdk
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Chemin vers le fichier JSON
file_path_com = r'c:\Users\axel.benoit\Documents\Demandes ponctuelles\Aurélie\com (2).json'
with open(file_path_com, 'r') as file:
    com = json.load(file)

df = pd.read_excel(r'c:\Users\axel.benoit\Documents\Demandes ponctuelles\Aurélie\flow_test.xlsx')
df['ORIGINE_CODE'] = df['ORIGINE_CODE'].astype(str)

WHITE_RGB = [255, 255, 255, 255]  # RGBA

com_codes_from_df = df['ORIGINE_CODE'].unique()
filtered_com = {
    "type": "FeatureCollection",
    "features": [
        feature for feature in com.get('features', [])
        if feature.get('properties', {}).get('insee') in com_codes_from_df
    ]
}


grouped_df = df.groupby(['ORIGINE_CODE', 'ORIGINE_NAME', 'DESTINATION_CODE', 'DESTINATION_NAME']).agg({
    'FLUX': 'sum',
    'X_START': 'first',
    'Y_START': 'first',
    'X_END': 'first',
    'Y_END': 'first'
}).reset_index()

# Trier les communes par flux et calculer la part de chaque commune
# Ajouter la valeur minimum et maximum pour définir les classes correctement
sorted_df = grouped_df.sort_values(by='FLUX', ascending=False)
sorted_df['LOG_VALUE%'] = np.log1p(sorted_df['FLUX'])
# Définir la rampe de couleur
cmap = plt.get_cmap('viridis')

# Normaliser les valeurs de flux pour qu'elles soient entre 0 et 1
norm = mcolors.Normalize(vmin=sorted_df['LOG_VALUE%'].min(), vmax=sorted_df['LOG_VALUE%'].max())

def color_scale(val):
        rgba = cmap(norm(val))
        return [int(255 * c) for c in rgba[:3]] + [127]  # Convertir en RGBA
    
sorted_df['color'] = sorted_df['LOG_VALUE%'].apply(color_scale)
color_mapping = dict(zip(sorted_df['ORIGINE_CODE'], sorted_df['color']))
for feature in filtered_com['features']:
    insee_code = feature['properties']['insee']
    feature['properties']['color'] = color_mapping.get(insee_code, [200, 200, 200, 100])


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

view_state = pdk.ViewState(
    latitude=44.837789,
    longitude=-0.57918,
    bearing=0,
    pitch=0,
    zoom=9,
    min_zoom=6,
)


r = pdk.Deck(
    layers=[com_layer],
    initial_view_state=view_state,
)

r.to_html("geojson_layer.html")

# %%
import pydeck as pdk
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Chemin vers le fichier JSON
file_path_com = r'c:\Users\axel.benoit\Documents\Demandes ponctuelles\Aurélie\com (2).json'
with open(file_path_com, 'r') as file:
    com = json.load(file)

df = pd.read_excel(r'c:\Users\axel.benoit\Documents\Demandes ponctuelles\Aurélie\flow_test.xlsx')
df['ORIGINE_CODE'] = df['ORIGINE_CODE'].astype(str)

WHITE_RGB = [255, 255, 255, 255]  # RGBA

com_codes_from_df = df['ORIGINE_CODE'].unique()
filtered_com = {
    "type": "FeatureCollection",
    "features": [
        feature for feature in com.get('features', [])
        if feature.get('properties', {}).get('insee') in com_codes_from_df
    ]
}


grouped_df = df.groupby(['ORIGINE_CODE', 'ORIGINE_NAME', 'DESTINATION_CODE', 'DESTINATION_NAME']).agg({
    'FLUX': 'sum',
    'X_START': 'first',
    'Y_START': 'first',
    'X_END': 'first',
    'Y_END': 'first'
}).reset_index()

# Trier les communes par flux et calculer la part de chaque commune
sorted_df = grouped_df.sort_values(by='FLUX', ascending=False)
sorted_df['LOG_VALUE%'] = np.log1p(sorted_df['FLUX'])

# Définir la rampe de couleur
cmap = plt.get_cmap('viridis')

# Normaliser les valeurs de flux pour qu'elles soient entre 0 et 1
norm = mcolors.Normalize(vmin=sorted_df['LOG_VALUE%'].min(), vmax=sorted_df['LOG_VALUE%'].max())

def color_scale(val):
    rgba = cmap(norm(val))
    return [int(255 * c) for c in rgba[:3]] + [127]  # Convertir en RGBA

sorted_df['color'] = sorted_df['LOG_VALUE%'].apply(color_scale)
color_mapping = dict(zip(sorted_df['ORIGINE_CODE'], sorted_df['color']))

for feature in filtered_com['features']:
    insee_code = feature['properties']['insee']
    feature['properties']['color'] = color_mapping.get(insee_code, [200, 200, 200, 100])

# Créez un dictionnaire de correspondance entre les codes INSEE et les valeurs de flux
flux_mapping = dict(zip(df['ORIGINE_CODE'], grouped_df['FLUX']))

# Ajoutez la valeur de flux à chaque feature
for feature in filtered_com['features']:
    insee_code = feature['properties']['insee']
    feature['properties']['flux'] = flux_mapping.get(insee_code, 0)  # Ajoutez la valeur de flux ou 0 si non trouvé

# Vérifiez les propriétés des features
for feature in filtered_com['features']:
    print(feature['properties'])

com_layer = pdk.Layer(
    "GeoJsonLayer",
    data=filtered_com,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=True,
    get_elevation='properties.flux',
    get_fill_color='properties.color',
    get_line_color=WHITE_RGB,
    get_line_width=100
)

view_state = pdk.ViewState(
    latitude=44.837789,
    longitude=-0.57918,
    bearing=0,
    pitch=0,
    zoom=9,
    min_zoom=6,
)

tooltip = {
    "html": "<b style='color: white;'>{flux}</b> Apprenant(s) issus de <b style='color: white;'>{nom}</b><br>"
}

r = pdk.Deck(
    layers=[com_layer],
    initial_view_state=view_state,
    map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    tooltip=tooltip
)

# Enregistrer le rendu en HTML
r.to_html("geojson_layer.html")

# %%



