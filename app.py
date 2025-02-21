# -*- coding: utf-8 -*-
"""
otccp project
Author: Dancan Oruko
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dask.dataframe as dd
from matplotlib import pyplot as plt
import numpy as np
import sqlite3
import os
import io
import base64
from scipy.stats import skew, kurtosis
import time
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from plotly import graph_objs as go
from sklearn.manifold import TSNE
from dash import callback_context


# Initialize the Dash app
app = dash.Dash(__name__)

# Database connection
DB_PATH = 'otccpdatabase.db'

# Create tables if they don't exist
def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS Analysis (
                        id INTEGER PRIMARY KEY,
                        analysis_name TEXT,
                        date TEXT,
                        num_cells INTEGER,
                        status TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS UploadedData (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        content TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS ProcessedData (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        content TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS MergedData (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        merged_filename TEXT,
                        data TEXT
                    )''')
    conn.commit()
    conn.close()

def reset_and_create_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing tables to reset
    cursor.execute("DROP TABLE IF EXISTS Analysis")
    cursor.execute("DROP TABLE IF EXISTS UploadedData")
    cursor.execute("DROP TABLE IF EXISTS ProcessedData")
    cursor.execute("DROP TABLE IF EXISTS MergedData")

    # Recreate tables
    cursor.execute('''CREATE TABLE Analysis (
                        id INTEGER PRIMARY KEY,
                        analysis_name TEXT,
                        date TEXT,
                        num_cells INTEGER,
                        status TEXT
                    )''')
    cursor.execute('''CREATE TABLE UploadedData (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        content TEXT
                    )''')
    cursor.execute('''CREATE TABLE ProcessedData (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        content TEXT
                    )''')
    cursor.execute('''CREATE TABLE MergedData (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        merged_filename TEXT,
                        data TEXT
                    )''')

    conn.commit()
    conn.close()
    
create_tables()

# App layout
app.layout = html.Div([
    # Top Bar
    html.Div([
        html.H1("BarrosoLab OTTCP Classification Dashboard", style={
            "textAlign": "center",
            "margin": "0",
            "padding": "20px 0",
            "color": "white",
            "fontSize": "24px"
        }),
    ], style={
        "backgroundColor": "#2C3E50",
        "padding": "10px 0",
        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"
    }),

    # Main Layout
    html.Div([
        # Sidebar
        html.Div([
            html.H2("Navigation", style={
                "textAlign": "center",
                "color": "#2C3E50",
                "marginBottom": "20px"
            }),

            html.Div([
                html.H4("File Upload", style={"color": "#34495E", "marginTop": "20px"}),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px 0',
                        'backgroundColor': '#ECF0F1', 'color': '#2C3E50'
                    },
                    multiple=True
                ),
                html.Div(id='uploaded-files-list'),

                html.H4("Initiate Secondary Processing", style={"color": "#34495E", "marginTop": "20px"}),
                html.Button("Run OTCCP Processing", id="processing-button", n_clicks=0, style={
                    "width": "100%", "marginBottom": "10px", "backgroundColor": "#E74C3C", "color": "white", "border": "none", "padding": "10px", "borderRadius": "5px"
                }),
                dcc.Loading(
                    id="loading-bar",
                    type="default",
                    children=[html.Div(id="processing-output")]
                ),

                html.H4("Merge Files", style={"color": "#34495E", "marginTop": "20px"}),
                html.Button("Merge Processed Files", id="merge-button", n_clicks=0, style={
                    "width": "100%", "marginBottom": "10px", "backgroundColor": "#1ABC9C", "color": "white", "border": "none", "padding": "10px", "borderRadius": "5px"
                }),
                html.Div(id="merge-output"),

                html.H4("View Data", style={"color": "#34495E", "marginTop": "20px"}),
                html.Button("clear database", id="clear-database", n_clicks=0, style={
                    "width": "100%", "marginBottom": "5px", "backgroundColor": "#2980B9", "color": "white", "border": "none", "padding": "10px", "borderRadius": "5px"
                }),
                html.Button("View Merged Data", id="view-merged-button", n_clicks=0, style={
                    "width": "100%", "backgroundColor": "#2980B9", "color": "white", "border": "none", "padding": "10px", "borderRadius": "5px"
                }),
            ], style={"padding": "20px"}),

            html.H4("Build Your Pivot Table", style={"color": "#34495E", "marginTop": "20px"}),

            html.Div([
                dcc.Dropdown(id='pivot-index', multi=False, placeholder='Select Index Column'),
                dcc.Dropdown(id='pivot-values', multi=False, placeholder='Select Values Column'),
                dcc.Dropdown(
                    id='pivot-aggfunc',
                    options=[
                        {'label': 'Mean', 'value': 'mean'},
                        {'label': 'Sum', 'value': 'sum'},
                        {'label': 'Count', 'value': 'count'},
                        {'label': 'Max', 'value': 'max'},
                        {'label': 'Min', 'value': 'min'}
                    ],
                    multi=False,
                    placeholder='Select Aggregation Function'
                ),
                html.Button("Generate Pivot Table", id="generate-pivot-button", n_clicks=0, style={
                    "width": "100%", "marginTop": "10px", "backgroundColor": "#1ABC9C", "color": "white",
                    "border": "none", "padding": "10px", "borderRadius": "5px"
                })
            ], style={"padding": "10px 0"}),




        ], style={
            "flex": "1",
            "backgroundColor": "#ECF0F1",
            "boxShadow": "4px 0 8px rgba(0, 0, 0, 0.1)"
        }),

        # Main Dashboard Area
        html.Div([

            html.Div([

                # Scrollable Card for DataTable
                html.Div([
                    dash_table.DataTable(
                        id='merged-data-table',
                        style_table={
                            'overflowY': 'auto',
                            'maxHeight': '400px',
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'whiteSpace': 'normal',
                            'height': 'auto'
                        },
                        page_size=10
                    ),

                    # Download Button
                    html.Div([
                        html.Button("Download Merged Data", id="download-button", n_clicks=0, style={
                            "backgroundColor": "#2980B9",
                            "color": "white",
                            "padding": "10px",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer"
                        }),
                        dcc.Download(id="download-dataframe-csv")
                    ], style={
                        "textAlign": "right",
                        "marginBottom": "20px"
                    }),

                ], style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "margin": "20px",
                    "boxShadow": "0px 4px 8px rgba(0,0,0,0.1)",
                    "borderRadius": "10px",
                    "overflow": "hidden",
                    "maxHeight": "50vh",
                    "overflowY": "scroll",
                    "marginBottom": "20px",
                    "width":"100%"
                }),

          


            ], style ={
                "display": "flex",
                "flexDirection": "row",   
              }),




                html.Div(id='data-table-container'),

                html.Div([
                    html.H4("Generated Pivot Table", style={"color": "#34495E"}),
                    dash_table.DataTable(id='pivot-table', style_table={'overflowY': 'auto', 'maxHeight': '400px'}),
                ], style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "boxShadow": "0px 4px 8px rgba(0,0,0,0.1)",
                    "borderRadius": "10px",
                    "overflow": "hidden",
                    "marginBottom": "20px"
                }),

                # Random Forest Model Training Section
                html.H4("Train Random Forest Model", style={"color": "#34495E", "marginTop": "20px"}),

                

                # Dropdown for selecting X (features)
         
                html.Div([
                    html.H4("Select Features for X", style={"color": "#34495E", "marginTop": "20px"}),

                    # Dropdown for selecting individual features
                    dcc.Dropdown(
                        id='rf-feature-selector',
                        multi=True,
                        placeholder="Select features for X",
                        style={"marginBottom": "10px"}
                    ),

                    # Button to select all features at once
                    html.Div([
                        html.Button("Select All Features", id="select-all-features-button", n_clicks=0, style={
                            "backgroundColor": "#E67E22",
                            "color": "white",
                            "padding": "10px",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "marginBottom": "20px"
                        }),
                        html.Button("Select DPG Features", id="select-dpg-features-button", n_clicks=0, style={
                            "backgroundColor": "#E67E22",
                            "color": "white",
                            "padding": "10px",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "marginBottom": "20px"
                        }),
                        html.Button("Select NPG Features", id="select-npg-features-button", n_clicks=0, style={
                            "backgroundColor": "#E67E22",
                            "color": "white",
                            "padding": "10px",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "marginBottom": "20px"
                        }),
                        html.Button("Select OPG Features", id="select-opg-features-button", n_clicks=0, style={
                            "backgroundColor": "#E67E22",
                            "color": "white",
                            "padding": "10px",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "marginBottom": "20px"
                        }),

                        # Button to deselect all features
                        html.Button(
                            "Deselect All Features",
                            id="deselect-all-features-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#C0392B",
                                "color": "white",
                                "padding": "10px",
                                "border": "none",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "marginBottom": "20px"
                            }
                        ),

                    ], style={"display": "flex", "gap": "10px"}),

                ], style={"padding": "10px 0"}),

                html.Div(id='dummy-output', style={'display': 'none'}),

                # Dropdown for selecting y (label)
                html.Div([
                    dcc.Dropdown(
                        id='rf-label-selector',
                        placeholder="Select label for y",
                        style={"marginBottom": "10px"}
                    ),
                ], style={"padding": "10px 0"}),

                # Train Button
                html.Button(
                    "Train Random Forest Model",
                    id="train-rf-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#1ABC9C",
                        "color": "white",
                        "padding": "10px",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "marginBottom": "10px"
                    }
                ),

                # Output Section for Accuracy and Feature Importance Plot
                html.Div([
                    html.H4("Model Accuracy:", id="rf-accuracy-output", style={"color": "#34495E"}),
                    dcc.Graph(id="rf-feature-importance-plot"),
                ], style={"marginTop": "20px"}),

                html.Div([

                    # Confusion Matrix Plot
                    html.Div([
                        dcc.Graph(id="rf-confusion-matrix-plot"),
                    ], style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "margin": "20px",
                        "boxShadow": "0px 4px 8px rgba(0,0,0,0.1)",
                        "borderRadius": "10px",
                        "marginBottom": "20px",
                        "maxWidth":"45%"
                    }),

                    # t-SNE Plot
                    html.Div([
                        dcc.Graph(id="rf-tsne-plot"),
                    ], style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "margin": "20px",
                        "boxShadow": "0px 4px 8px rgba(0,0,0,0.1)",
                        "borderRadius": "10px",
                        "marginBottom": "20px",
                        "maxWidth":"45%"
                    }),
                ], style  = {
                    "display":"flex",
                    "flexDirection":"row"

                })



                

        ], style={
            "flex": "4",
            "padding": "20px",
            "backgroundColor": "#F8F9F9",
            "maxWidth":"80%"
        })
    ], style={
        "display": "flex",
        "flexDirection": "row",
    })

])

# Secondary OTCCP Processing Function using Dask
def secondary_processing(df):
    figures = df.compute().values
    featureList = df.columns.tolist()
    
    processed_df = pd.DataFrame()
    processed_df["area"] = np.array([row[featureList.index('Area')] for row in figures])
    processed_df["volume"] = np.array([row[featureList.index('Volume')] for row in figures])
    processed_df["bbaa_x"] = np.array([row[featureList.index('BoundingBoxAA Length X')] for row in figures])
    processed_df["bbaa_y"] = np.array([row[featureList.index('BoundingBoxAA Length Y')] for row in figures])
    processed_df["bbaa_z"] = np.array([row[featureList.index('BoundingBoxAA Length Z')] for row in figures])
    processed_df["bboo_x"] = np.array([row[featureList.index('BoundingBoxOO Length A')] for row in figures])
    processed_df["bboo_y"] = np.array([row[featureList.index('BoundingBoxOO Length B')] for row in figures])
    processed_df["bboo_z"] = np.array([row[featureList.index('BoundingBoxOO Length C')] for row in figures])
    processed_df["pos_x"] = np.array([row[featureList.index('Position X Reference Frame')] for row in figures])
    processed_df["pos_y"] = np.array([row[featureList.index('Position Y Reference Frame')] for row in figures])
    processed_df["pos_z"] = np.array([row[featureList.index('Position Z Reference Frame')] for row in figures])
    processed_df["Distance from Origin"] = np.array([row[featureList.index('Distance from Origin')] for row in figures])
    processed_df["Shortest Distance to Surfaces"] = np.array([row[featureList.index('Position Z')] for row in figures])
    processed_df["Triangles"] = np.array([row[featureList.index('Number of Triangles')] for row in figures])
    processed_df["Voxels"] = np.array([row[featureList.index('Number of Voxels')] for row in figures])
    processed_df["Sphericity"] = np.array([row[featureList.index('Sphericity')] for row in figures])
    processed_df["Prolate"] = np.array([row[featureList.index('Ellipticity (prolate)')] for row in figures])
    processed_df["Oblate"] = np.array([row[featureList.index('Ellipticity (oblate)')] for row in figures])

    processed_df = processed_df.dropna(how = "all")
    
    num_rows = processed_df["area"].size
    warnings.warn(f"{num_rows}")
    objectID = np.arange(num_rows).reshape(-1, 1).ravel().tolist()



    cellnum = objectID.count(0)
    mmdist_data = []
    posx_idx = featureList.index("Position X Reference Frame")
    posy_idx = featureList.index("Position Y Reference Frame")
    posz_idx = featureList.index("Position Z Reference Frame")
    

    start_index = [index for index, value in enumerate(objectID) if value == 0]
    

    for i in range(cellnum):
        cellstart = start_index[i]
        if i + 1 < cellnum:
            cellend = start_index[i + 1] - 1
        else:
            cellend = len(objectID) - 1

        cellrange = np.arange(cellstart, cellend + 1)
        current_mmdist_data = np.zeros((len(cellrange), len(cellrange)))

        for x in range(len(cellrange)):
            for z in range(len(cellrange)):
                if x == z:
                    current_mmdist_data[x, z] = 0
                    continue

                obj1_idx = cellrange[x]
                obj2_idx = cellrange[z]
                obj1_pos = [figures[obj1_idx][posx_idx], figures[obj1_idx][posy_idx], figures[obj1_idx][posz_idx]]
                obj2_pos = [figures[obj2_idx][posx_idx], figures[obj2_idx][posy_idx], figures[obj2_idx][posz_idx]]

                mm_dist = np.sqrt((obj1_pos[0] - obj2_pos[0]) ** 2 +
                                  (obj1_pos[1] - obj2_pos[1]) ** 2 +
                                  (obj1_pos[2] - obj2_pos[2]) ** 2)

                current_mmdist_data[x, z] = mm_dist

        mmdist_data.append(current_mmdist_data)


    min_mmdist, max_mmdist, mean_mmdist, median_mmdist = [], [], [], []
    std_mmdist, sum_mmdist, skewness_mmdist, kurtosis_mmdist = [], [], [], []
    
    for current_mmdist_data in mmdist_data:
        for j in range(len(current_mmdist_data)):
            obj_mmdist = current_mmdist_data[j, current_mmdist_data[j, :] != 0]
            min_mmdist.append(np.min(obj_mmdist))
            max_mmdist.append(np.max(obj_mmdist))
            mean_mmdist.append(np.mean(obj_mmdist))
            median_mmdist.append(np.median(obj_mmdist))
            std_mmdist.append(np.std(obj_mmdist, ddof=1))
            sum_mmdist.append(np.sum(obj_mmdist))
            skewness_mmdist.append(skew(obj_mmdist))
            kurtosis_mmdist.append(kurtosis(obj_mmdist, fisher=False))
    
    
    processed_df['Object ID'] =  objectID
    processed_df['Min Dist'] =  min_mmdist
    processed_df['Max Dist'] =  max_mmdist
    processed_df['Mean Dist'] =  mean_mmdist
    processed_df['Median Dist'] =  median_mmdist
    processed_df['Std Dist'] =  std_mmdist
    processed_df['Sum Dist'] =  sum_mmdist
    processed_df['Skewness'] =  skewness_mmdist
    processed_df['Kurtosis Dist'] =  kurtosis_mmdist

    

    # Truncate the "label" column to match the length of "Oblate"
    processed_df["Nucleus"] = np.array([row[featureList.index('label')] for row in figures])[:len(processed_df["Oblate"])]
    nucleus_values = processed_df["Nucleus"][0]
    processed_df["Nucleus"] = np.resize(nucleus_values, num_rows)
    processed_df['label'] = processed_df['Nucleus'].str.split(' ').str[0]
    processed_df['cell_number'] = processed_df['Nucleus'].str.extract(r'Cell (\d+)')


    

    return processed_df

# Callback to handle file upload
@app.callback(
    Output('uploaded-files-list', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_file_upload(contents, filenames):
    if contents is None:
        return html.Div()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    children = []

    for content, filename in zip(contents, filenames):
        try:
            content_type, content_string = content.split(',')
            cursor.execute('INSERT INTO UploadedData (filename, content) VALUES (?, ?)', (filename, content_string))
            children.append(html.Div([f'File {filename} uploaded successfully!']))
        except Exception as e:
            children.append(html.Div([f'Error uploading file {filename}: {str(e)}']))

    conn.commit()
    conn.close()
    return children

# Callback to initiate secondary processing
@app.callback(
    Output('processing-output', 'children'),
    Input('processing-button', 'n_clicks')
)
def run_secondary_processing(n_clicks):
    if n_clicks == 0:
        return html.Div()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT filename, content FROM UploadedData')
        files = cursor.fetchall()
          

        warnings.warn("This is a warning message!")

        if not files:
            return html.Div("No files to process.")

        for filename, content_string in files:
            decoded = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))
            decoded = dd.from_pandas(decoded,npartitions=2) 

            processed_df = secondary_processing(decoded)
            
            warnings.warn("This is a warning message!")
            processed_df = dd.from_pandas(processed_df, npartitions = 2)
           
            processed_data_text = processed_df.compute().to_csv(index=False)
           
            cursor.execute('INSERT INTO ProcessedData (filename, content) VALUES (?, ?)', (filename, processed_data_text))
            

        conn.commit()
        return html.Div("Secondary processing completed successfully!")

    except Exception as e:
        return html.Div(f"Error during processing: {str(e)}")

    finally:
        conn.close()

# Callback to merge processed files
@app.callback(
    Output('merge-output', 'children'),
    Input('merge-button', 'n_clicks')
)
def merge_files(n_clicks):
    if n_clicks == 0:
        return html.Div()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT content FROM ProcessedData')
        files = cursor.fetchall()

        if not files:
            return html.Div("No processed files to merge.")

        dfs = [pd.read_csv(io.StringIO(file[0])) for file in files]
        merged_df = pd.concat(dfs, axis=0)
        
        #merged_df = dd.from_pandas(merged_df, npartitions=3) 

        merged_filename = "merged_data.csv"
        merged_data_text = merged_df.to_csv(index=False)
        cursor.execute('INSERT INTO MergedData (merged_filename, data) VALUES (?, ?)', (merged_filename, merged_data_text))

        conn.commit()
        return html.Div("Files merged successfully and stored in the database!")

    except Exception as e:
        return html.Div(f"Error merging files: {str(e)}")

    finally:
        conn.close()

# Callback to display merged data in a table
@app.callback(
    Output('merged-data-table','data'),
    Input('view-merged-button', 'n_clicks'),
    prevent_initial_call=True
)


def display_merged_data(n_clicks):
    if n_clicks == 0:
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT data FROM MergedData ORDER BY id DESC LIMIT 1')
        merged_data = cursor.fetchone()

        if merged_data is None:
            return []

        merged_df = pd.read_csv(io.StringIO(merged_data[0]))
        return merged_df.to_dict('records')

    except Exception as e:
        return []

    finally:
        conn.close()


# # Callback to display merged data in a table
@app.callback(
        
    Output('dummy-output', "children"),
    Input('clear-database', 'n_clicks')
)


def clear_data(n_clicks):
    try:
        reset_and_create_tables()
        return []
        
    except Exception as e:
        return []

   

@app.callback(
    [Output('pivot-index', 'options'),
     Output('pivot-values', 'options')],
    Input('view-merged-button', 'n_clicks')
)
def populate_pivot_dropdowns(n_clicks):
    if n_clicks == 0:
        return [], []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT data FROM MergedData ORDER BY id DESC LIMIT 1')
        merged_data = cursor.fetchone()
        if not merged_data:
            return [], []

        merged_df = pd.read_csv(io.StringIO(merged_data[0]))
        column_options = [{'label': col, 'value': col} for col in merged_df.columns]
        return column_options, column_options

    except Exception as e:
        return [], []

    finally:
        conn.close()


@app.callback(
    [Output('pivot-table', 'data'),
     Output('pivot-table', 'columns')],
    [Input('generate-pivot-button', 'n_clicks')],
    [State('pivot-index', 'value'),
     State('pivot-values', 'value'),
     State('pivot-aggfunc', 'value'),
     State('view-merged-button', 'n_clicks')]
)
def generate_pivot_table(n_clicks, index_col, values_col, agg_func, view_clicks):
    if n_clicks == 0 or not index_col or not values_col or not agg_func:
        return [], []

    # Load merged data from the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT data FROM MergedData ORDER BY id DESC LIMIT 1')
        merged_data = cursor.fetchone()
        if not merged_data:
            return [], []

        merged_df = pd.read_csv(io.StringIO(merged_data[0]))

        # Generate the pivot table
        pivot_table = pd.pivot_table(merged_df, index=index_col, values=values_col, aggfunc=agg_func)

        # Convert pivot table to DataTable format
        data = pivot_table.reset_index().to_dict('records')
        columns = [{'name': col, 'id': col} for col in pivot_table.reset_index().columns]

        return data, columns

    except Exception as e:
        return [], []

    finally:
        conn.close()



@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_merged_data(n_clicks):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # try:
    # Retrieve merged data from the database
    cursor.execute('SELECT data FROM MergedData ORDER BY id DESC LIMIT 1')
    merged_data = cursor.fetchone()

    if not merged_data:
        return dcc.send_string("No data available for download.", filename="empty.csv")

    # Convert merged data to DataFrame
    merged_df = pd.read_csv(io.StringIO(merged_data[0]))

    # Return the DataFrame as a excel file
    return dcc.send_data_frame(merged_df.to_excel, "merged_data.xlsx", sheet_name="MergedData")

    # except Exception as e:
    #     return dcc.send_string(f"Error: {str(e)}", filename="error.csv")

    # finally:
    # conn.close()



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
from plotly import graph_objs as go
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import io
import base64

@app.callback(
    [Output("rf-accuracy-output", "children"),
     Output("rf-feature-importance-plot", "figure"),
     Output("rf-confusion-matrix-plot", "figure"),
     Output("rf-tsne-plot", "figure")],
    Input("train-rf-button", "n_clicks"),
    State("rf-feature-selector", "value"),
    State("rf-label-selector", "value"),
    Input('view-merged-button', 'n_clicks')
)
def train_random_forest(n_clicks, feature_columns, label_column, view_clicks):
    if n_clicks == 0 or not feature_columns or not label_column:
        return "Please select features and label.", {}, {}, {}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Retrieve merged data
        cursor.execute('SELECT data FROM MergedData ORDER BY id DESC LIMIT 1')
        merged_data = cursor.fetchone()
        if not merged_data:
            return "No data available.", {}, {}, {}

        merged_df = pd.read_csv(io.StringIO(merged_data[0]))

        # Split features and label
        X = merged_df[feature_columns]
        y = merged_df[label_column].values  # Ensure y is a NumPy array

        # Cross-validation
        model = RandomForestClassifier(random_state=42, n_jobs=1)
        cv = StratifiedKFold(n_splits=5)
        scores = cross_val_score(model, X, y, cv=cv)
        accuracy = scores.mean()

        # Cross-validation predictions
        y_pred = cross_val_predict(model, X, y, cv=cv)

        # Confusion Matrix using Plotly
        conf_matrix = confusion_matrix(y, y_pred)
        conf_matrix_norm = (conf_matrix / np.sum(conf_matrix)) * 100
        z_text = [[f"{val:.2f}%" for val in row] for row in conf_matrix_norm]

        conf_matrix_fig = ff.create_annotated_heatmap(
            z=conf_matrix_norm,
            x=list(np.unique(y)),
            y=list(np.unique(y)),
            annotation_text=z_text,
            colorscale="Blues"
        )
        conf_matrix_fig.update_layout(
            title="Confusion Matrix (Cross-Validated)",
            xaxis_title="Predicted Labels",
            yaxis_title="True Labels"
        )

        # Train the model for feature importance and t-SNE
        model.fit(X, y)
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)
        feature_fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance")


        # Compute Proximity Matrix for t-SNE
        leaves = model.apply(X)
        proximity = compute_proximity(leaves)
        distance = 1 - proximity

       

        # Perform t-SNE
        t_sne = TSNE(n_components=2, perplexity=20, metric="precomputed", init="random", verbose=1, learning_rate="auto").fit_transform(distance)

        
        # Generate hover labels
        def label_fn(row, label_value):
            cell_id = row[0]  # row[0] contains the index (cell ID)
            row_data = pd.Series(row[1])  # Convert row[1] to a Pandas Series
            label_info = "<br>".join([f"<b>{col}:</b> {val}" for col, val in row_data.items()])
            return f"<b>Cell ID:</b> {cell_id}<br><b>Label:</b> {label_value}<br>{label_info}"

        # Use zip to iterate through both X and y
        labels = [label_fn(row, label) for row, label in zip(X.iterrows(), y)]


        # Interactive t-SNE Plot
        unique_labels = np.unique(y)
        colors = px.colors.qualitative.Safe[:len(unique_labels)]
        label_to_color = {int(label): color for label, color in zip(unique_labels, colors)}  # Ensure labels are scalar

        tsne_traces = []
        for label, color in label_to_color.items():
            indices = np.where(y == label)[0]
            tsne_traces.append(go.Scatter(
                x=t_sne[indices, 0],
                y=t_sne[indices, 1],
                mode='markers',
                name=str(label),
                marker=dict(color=color, size=6, opacity=0.8),
                text=[label_fn(X.iloc[i], label) for i in indices]
            ))

        tsne_fig = go.Figure(data=tsne_traces)
        tsne_fig.update_layout(
            title="Interactive t-SNE Plot",
            template="simple_white",
            hovermode="closest",
            legend_title="Class Labels"
        )

        return f"Cross-Validated Accuracy: {accuracy:.2%}", feature_fig, conf_matrix_fig, tsne_fig

    except Exception as e:
        return f"Error: {str(e)}", {}, {}, {}

    finally:
        conn.close()


@app.callback(
    [Output('rf-feature-selector', 'options'),
     Output('rf-label-selector', 'options')],
    Input('view-merged-button', 'n_clicks')
)
def populate_rf_dropdowns(n_clicks):
    if n_clicks == 0:
        return [], []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT data FROM MergedData ORDER BY id DESC LIMIT 1')
        merged_data = cursor.fetchone()

        if not merged_data:
            return [], []

        merged_df = pd.read_csv(io.StringIO(merged_data[0]))
        column_options = [{'label': col, 'value': col} for col in merged_df.columns]

        return column_options, column_options

    except Exception as e:
        return [], []

    finally:
        conn.close()


def compute_proximity(leaves, step_size=100):
    """Computes the proximity between each pair of examples."""
    example_idx = 0
    num_examples = leaves.shape[0]
    t_leaves = np.transpose(leaves)
    proximities = []

    while example_idx < num_examples:
        end_idx = min(example_idx + step_size, num_examples)
        proximities.append(
            np.mean(
                leaves[..., np.newaxis] == t_leaves[:, example_idx:end_idx][np.newaxis, ...],
                axis=1
            )
        )
        example_idx = end_idx

    return np.concatenate(proximities, axis=1)

def perform_tsne(proximity):
    """Performs t-SNE on the proximity matrix."""
    distance = 1 - proximity
    t_sne = TSNE(n_components=2, perplexity=20, metric="precomputed", init="random", verbose=1, learning_rate="auto").fit_transform(distance)
    return t_sne




@app.callback(
    Output('rf-feature-selector', 'value'),
    [Input('select-all-features-button', 'n_clicks'),
     Input('select-dpg-features-button', 'n_clicks'),
     Input('select-npg-features-button', 'n_clicks'),
     Input('select-opg-features-button', 'n_clicks'),
     Input('deselect-all-features-button', 'n_clicks')],
    [State('rf-feature-selector', 'options'),
     State('rf-feature-selector', 'value')]
)
def update_feature_selection(select_all_clicks, select_dpg_clicks, select_npg_clicks, select_opg_clicks, deselect_all_clicks, feature_options, current_selection):
    # Identify which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return current_selection

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Feature groups
    dpg_features = [
        "Kurtosis Dist", "Mean Dist", "Skewness", "Sum/Vol Dist", "Sum/Pos Z Dist",
        "Sum*Pos Z Dist", "Sum*Vol Dist", "Max Dist", "Median Dist", "Min Dist",
        "Std Dist", "Sum Dist"
    ]
    npg_features = ["Polarity", "Position X", "Position Y", "Position Z", "Dist Origin", "Dist Surface"]
    opg_features = ["Area", "BB AA X", "BB AA Y", "BB AA Z", "BB OO X", "BB OO Y", "BB OO Z",
                    "BI AA", "BI OO", "CI", "Oblate", "Prolate", "Sphericity", "Triangles", "Volume", "Voxels"]

    # If no features are currently selected, initialize the selection
    if current_selection is None:
        current_selection = []

    # Handle "Select All Features" button
    if button_id == 'select-all-features-button':
        return list(set(current_selection + [option['value'] for option in feature_options]))

    # Handle "Select DPG Features" button
    elif button_id == 'select-dpg-features-button':
        return list(set(current_selection + [option['value'] for option in feature_options if option['value'] in dpg_features]))

    # Handle "Select NPG Features" button
    elif button_id == 'select-npg-features-button':
        return list(set(current_selection + [option['value'] for option in feature_options if option['value'] in npg_features]))

    # Handle "Select OPG Features" button
    elif button_id == 'select-opg-features-button':
        return list(set(current_selection + [option['value'] for option in feature_options if option['value'] in opg_features]))

    # Handle "Deselect All Features" button
    elif button_id == 'deselect-all-features-button':
        return []

    return current_selection


