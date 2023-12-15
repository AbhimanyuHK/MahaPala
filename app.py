"""

Created by abhimanyu at 15/12/23

"""

import base64
from io import BytesIO

import dash
import dash_bootstrap_components as dbc
from PIL import Image
from dash import html, dcc, callback, Output, Input, State

from mahapala.fruits_classification import FruitsClassification

fc = FruitsClassification()

app = dash.Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

navbar = dbc.NavbarSimple(
    brand="MahaPala",
    brand_href="#",
    color="primary",
    dark=True,
)

row = html.Div(
    [
        dbc.Row(
            [
                dbc.Col([
                    html.Br(),
                    dbc.Alert("Upload a fruit / plant to identify", style={"text-align": "center"}),
                    html.Br(),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    html.Div(id='output-data-upload'),
                ]),
                dbc.Col([
                    html.Br(),
                    dbc.Alert("Predication Result", color="info", style={"text-align": "center"}),
                    html.Br(),
                    dcc.Loading(
                        id="loading-input-1",
                        type="default",
                        children=html.Div(id="loading-output-1")
                    ),
                    html.Div(id='output-data-prediction-result'),
                ]),
            ]
        ),
    ]
)

app.layout = html.Div([
    navbar,
    row
])


@callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    print(list_of_contents, list_of_names, list_of_dates)
    if list_of_names:
        return dbc.Alert(
            [
                html.H4(list_of_names, className="alert-heading"),
                html.Hr(),
                html.P(
                    html.Img(src=list_of_contents),
                    className="mb-0",
                ),
            ]
        )


@callback(
    [Output('output-data-prediction-result', 'children'), Output("loading-output-1", "children")],
    [Input('upload-data', 'contents')],
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output_predication_result(list_of_contents, list_of_names, list_of_dates):
    print(list_of_contents, list_of_names, list_of_dates)
    if list_of_names:
        img = Image.open(BytesIO(base64.b64decode(list_of_contents.split(",")[1])))
        img = img.resize((224, 224))
        # img.save('input_request.jpg')
        fc.fit()
        result = fc.predict_image(img)

        lt = []
        for x in result:
            progress_value = float(x["confidence"].split(" ")[0])
            progress = dbc.Progress(label=f"{progress_value}%", value=progress_value)

            class_div = html.Div(f"Class : {x['class']}")
            lt.append(
                html.Div([
                    class_div,
                    html.Br(),
                    progress
                ])
            )

        return dbc.Alert(
            [
                html.H4(list_of_names, className="alert-heading"),
                html.Hr(),
                html.P(
                    html.Div(lt),
                    className="mb-0",
                ),
            ],
            color="info",
        ), ""
    return "", ""


if __name__ == "__main__":
    app.run_server(debug=True)
