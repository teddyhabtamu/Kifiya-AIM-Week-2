import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

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
                    html.P("This dashboard provides insights into user engagement, experience, and satisfaction.", className="card-text"),
                    html.P("Use the navigation bar to explore the different analyses.", className="card-text")
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
    try:
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
    except Exception as e:
        return dbc.Container([
            html.H1("Error"),
            html.P(f"An error occurred: {e}")
        ])

# Define the home layout
def home_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Welcome to the Telecom Dashboard", className="card-title"),
                        html.P("This is the main page where you can navigate to various analyses using the menu above.", className="card-text")
                    ])
                ], color="info", outline=True)
            ], width=12),
        ], className="mb-4"),
    ])

# Define the user engagement layout
def user_engagement_layout():
    fig = px.scatter(df_engagement, x='session_duration', y='total_traffic', title='Session Duration vs Total Traffic')
    return dbc.Container([
        html.H3("User Engagement Analysis", className="mb-3"),
        html.P("This analysis explores the relationship between session duration and total data traffic.", className="lead"),
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
        html.P("This section compares the average throughput across different handset types to understand their performance.", className="lead"),
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
        html.P("This analysis explores the relationship between user engagement and overall satisfaction scores.", className="lead"),
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig)
            ])
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
