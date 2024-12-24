import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data
df_engagement = pd.read_csv('https://drive.google.com/file/d/1PRi1QZW6vOP9I13kpX4YamAXFOwCr-Nm/view?usp=sharing')
df_experience = pd.read_csv('https://drive.google.com/file/d/1wOVFHfEQn9uKoV7uGEqTPNvAg5-0AgfX/view?usp=drive_link')
df_satisfaction = pd.read_csv('https://drive.google.com/file/d/1P7ndGXOGoqDjmMUxGvc6QQtCEhx6H5Ja/view?usp=drive_link')

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
    dbc.Container(id='page-content', className="mt-4"),
    dcc.Location(id='url', refresh=False)
])

# Define the callback to update the page content
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
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

# Define the home layout
def home_layout():
    return dbc.Container([
        html.H1("Welcome to the Telecom Dashboard"),
        html.P("Use the navigation bar to explore different analyses.")
    ])

# Define the user overview layout
def user_overview_layout():
    fig = px.histogram(df_engagement, x='session_frequency', title='Session Frequency Distribution')
    return dbc.Container([
        html.H1("User Overview Analysis"),
        dcc.Graph(figure=fig)
    ])

# Define the user engagement layout
def user_engagement_layout():
    fig = px.scatter(df_engagement, x='session_duration', y='total_traffic', title='Session Duration vs Total Traffic')
    return dbc.Container([
        html.H1("User Engagement Analysis"),
        dcc.Graph(figure=fig)
    ])

# Define the experience analysis layout
def experience_analysis_layout():
    fig = px.box(df_experience, x='Handset Type', y='Average Throughput', title='Average Throughput per Handset Type')
    return dbc.Container([
        html.H1("Experience Analysis"),
        dcc.Graph(figure=fig)
    ])

# Define the satisfaction analysis layout
def satisfaction_analysis_layout():
    fig = px.scatter(df_satisfaction, x='engagement_score', y='experience_score', title='Engagement Score vs Experience Score')
    return dbc.Container([
        html.H1("Satisfaction Analysis"),
        dcc.Graph(figure=fig)
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
