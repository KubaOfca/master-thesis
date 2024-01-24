import base64
import io
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from PIL import Image
from ultralytics import YOLO
import dash
from dash.dependencies import Input, Output
from PIL import Image
import numpy as np

# Define Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(
    style={
        'background': 'linear-gradient(to right, #c3ddfd, #ffffff)',
        'minHeight': '100vh',
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'justifyContent': 'center',
        'padding': '20px'
    },
    children=[
        html.H1(
            "🌟 Image Classification App 🌟",
            style={'fontFamily': 'Arial, sans-serif', 'color': '#333333', 'textAlign': 'center'}
        ),
        dcc.Upload(
            id='upload-image',
            children=[
                '📁 Drag and Drop or ',
                html.A('Select Files', style={'textDecoration': 'underline'})
            ],
            multiple=False,
            style={
                'width': '100%',  # Set width to 100%
                'height': '150px',
                'lineHeight': '150px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '20px auto 10px'
            }
        ),
        html.Div(id='output-image-preview', style={'text-align': 'center', 'margin-top': '10px'}),
        html.Div(id='output-image-upload', style={'text-align': 'center', 'margin-top': '10px'}),
        html.Button(
            '🔮 Predict 🏃',  # Running emoji
            id='predict-button',
            n_clicks=0,
            style={
                'background': 'linear-gradient(to right, #4e9af1, #0066cc)',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'display': 'block',
                'margin-top': '20px'
            }
        ),
    ]
)

IMG_LOAD = None
def pil_to_b64(im, enc_format="png", **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """

    buff = io.BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded
# Callback to perform prediction and display the result
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('output-image-upload', 'style')],
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_output(image_content, image_filename):
    if image_content is not None:
        # Convert image content to image
        content_type, content_string = image_content.split(',')
        decoded_image = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded_image))

        global IMG_LOAD
        IMG_LOAD = image

        # Display the uploaded image
        img_element = html.Img(src=image_content, style={'width': '50%', 'margin-top': '10px'})

        return [
            [html.H5(f'📸 Filename: {image_filename}'), img_element],
            {'display': 'block', 'text-align': 'center'}
        ]
    else:
        return [None, {'display': 'none'}]

@app.callback(
    [Output('output-image-upload', 'children', allow_duplicate=True),
     Output('output-image-upload', 'style', allow_duplicate=True)],
    [Input('predict-button', 'n_clicks')],
    [State('upload-image', 'contents'),
     State('upload-image', 'filename')],
    prevent_initial_call=True

)
def predict_image(n_clicks, image_content, image_filename):
    if n_clicks > 0 and image_content is not None:
        global IMG_LOAD
        model = YOLO(r"C:\Users\Jakub Lechowski\Desktop\master-thesis\code\best.pt")
        predict__ = model.predict(IMG_LOAD, stream=False, save=False, verbose=False)
        im_array = predict__[0].plot()
        im = Image.fromarray(im_array[..., ::-1])
        return [
            html.Img(
                src=f'data:image/png;base64,{pil_to_b64(im)}',
                style={'width': '100%'}
            ),
            html.H5(f'📸 Filename: {im}'),

        ]
    else:
        return [None, {'display': 'none'}]


if __name__ == '__main__':
    app.run_server(debug=True)
