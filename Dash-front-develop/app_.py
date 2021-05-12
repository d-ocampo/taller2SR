
import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__,
                external_stylesheets=["./assets/boostrap.min.css"],
                # these meta_tags ensure content is scaled correctly on different devices
                # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                ],
                assets_external_path='./', 
                )
app.title = 'SR - LastFM 1k'

app.config.suppress_callback_exceptions=True
