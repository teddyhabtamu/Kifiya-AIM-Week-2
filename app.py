import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX],
                meta_tags=[
                    {"name": "apple-touch-icon", "sizes": "57x57", "href": "/assets/apple-icon-57x57.png"},
                    {"name": "apple-touch-icon", "sizes": "60x60", "href": "/assets/apple-icon-60x60.png"},
                    {"name": "apple-touch-icon", "sizes": "72x72", "href": "/assets/apple-icon-72x72.png"},
                    {"name": "apple-touch-icon", "sizes": "76x76", "href": "/assets/apple-icon-76x76.png"},
                    {"name": "apple-touch-icon", "sizes": "114x114", "href": "/assets/apple-icon-114x114.png"},
                    {"name": "apple-touch-icon", "sizes": "120x120", "href": "/assets/apple-icon-120x120.png"},
                    {"name": "apple-touch-icon", "sizes": "144x144", "href": "/assets/apple-icon-144x144.png"},
                    {"name": "apple-touch-icon", "sizes": "152x152", "href": "/assets/apple-icon-152x152.png"},
                    {"name": "apple-touch-icon", "sizes": "180x180", "href": "/assets/apple-icon-180x180.png"},
                    {"name": "icon", "type": "image/png", "sizes": "192x192", "href": "/assets/android-icon-192x192.png"},
                    {"name": "icon", "type": "image/png", "sizes": "32x32", "href": "/assets/favicon-32x32.png"},
                    {"name": "icon", "type": "image/png", "sizes": "96x96", "href": "/assets/favicon-96x96.png"},
                    {"name": "icon", "type": "image/png", "sizes": "16x16", "href": "/assets/favicon-16x16.png"},
                    {"name": "manifest", "href": "/assets/manifest.json"},
                    {"name": "msapplication-TileColor", "content": "#ffffff"},
                    {"name": "msapplication-TileImage", "content": "/assets/ms-icon-144x144.png"},
                    {"name": "theme-color", "content": "#ffffff"}
                ])

# Load data
df_engagement = pd.read_csv('https://drive.google.com/uc?id=1PRi1QZW6vOP9I13kpX4YamAXFOwCr-Nm')
df_experience = pd.read_csv('https://drive.google.com/uc?id=1wOVFHfEQn9uKoV7uGEqTPNvAg5-0AgfX')
df_satisfaction = pd.read_csv('https://drive.google.com/uc?id=1P7ndGXOGoqDjmMUxGvc6QQtCEhx6H5Ja')

# Define the layout
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Telecom Dashboard",
        brand_href="/",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("User Overview", href="/user-overview")),
            dbc.NavItem(dbc.NavLink("User Engagement", href="/user-engagement")),
            dbc.NavItem(dbc.NavLink("Experience Analysis", href="/experience-analysis")),
            dbc.NavItem(dbc.NavLink("Satisfaction Analysis", href="/satisfaction-analysis")),
        ]
    ),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Welcome to the Telecom Dashboard", className="card-title"),
                    html.P("Use the navigation bar to explore different analyses.", className="card-text")
                ])
            ], color="info", outline=True)
        ], width=12),
    ], className="mb-4"),
    dbc.Container(id='page-content', className="mt-4"),
    dcc.Location(id='url', refresh=False)
])

# Define the callback to update the page content
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/user-overview':
        return user_overview_layout()
    elif pathname == '/user-engagement':
        return user_engagement_layout()
    elif pathname == '/experience-analysis':
        return experience_analysis_layout()
    elif pathname == '/satisfaction-analysis':
        return satisfaction_analysis_layout()
    else:
        return home_layout()

# Define the user overview layout
def user_overview_layout():
    fig = px.histogram(df_engagement, x='session_frequency', title='Session Frequency Distribution')
    return dbc.Container([
        html.H3("User Overview Analysis", className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig)
            ])
        ])
    ])

# Define the user engagement layout
def user_engagement_layout():
    fig = px.scatter(df_engagement, x='session_duration', y='total_traffic', title='Session Duration vs Total Traffic')
    return dbc.Container([
        html.H3("User Engagement Analysis", className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig)
            ])
        ])
    ])

# Define the experience analysis layout
def experience_analysis_layout():
    fig = px.box(df_experience, x='Handset Type', y='Average Throughput', title='Average Throughput per Handset Type')
    return dbc.Container([
        html.H3("Experience Analysis", className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig)
            ])
        ])
    ])

# Define the satisfaction analysis layout
def satisfaction_analysis_layout():
    fig = px.scatter(df_satisfaction, x='engagement_score', y='experience_score', title='Engagement Score vs Experience Score')
    return dbc.Container([
        html.H3("Satisfaction Analysis", className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig)
            ])
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
