import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from data_processor import WeatherDataProcessor

# 1. Initialize Processor and Data
# We clean the data once when the server starts
df = pd.read_csv("weather_cleaned_for_dash.csv")
raw_df = pd.read_csv("weatherAUS.csv", usecols=['WindGustDir', 'RainTomorrow']) # Load only what's needed
df['Date'] = pd.to_datetime(df['Date'])

# 2. Prepare Global Visuals (Calculated once)
# Correlation Heatmap
corr = df.corr(numeric_only=True)
fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                     title="Feature Correlation Matrix", aspect="auto")

# Wind Rose (Using raw_df for original wind directions)
wind_data = raw_df.groupby(['WindGustDir', 'RainTomorrow']).size().reset_index(name='count')
fig_wind = px.bar_polar(wind_data, r="count", theta="WindGustDir", color="RainTomorrow",
                        template="plotly_dark", barmode="stack", 
                        title="Wind Direction Impact on Rain")

# 3. Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Australian Weather Analytics", className="text-center text-primary my-4"), width=12)
    ]),

    dbc.Tabs([
        # Tab 1: Overview & Correlations
        dbc.Tab(label="General Overview", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_corr), width=12),
            ], className="mt-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='rain-target-dist', 
                                  figure=px.histogram(df, x="RainTomorrow", color="RainTomorrow", 
                                                      title="Target Distribution (Imbalance Check)")), width=6),
                dbc.Col(dcc.Graph(figure=fig_wind), width=6),
            ])
        ]),

        # Tab 2: Feature Deep Dive (Boxplots & KDE)
        dbc.Tab(label="Feature Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Feature to Analyze:"),
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=[{'label': c, 'value': c} for c in ['Humidity3pm', 'Cloud_Total', 'Sunshine', 'Pressure_Diff', 'Pressure3pm']],
                        value='Humidity3pm'
                    )
                ], width=4, className="mt-4")
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='feature-boxplot'), width=6),
                dbc.Col(dcc.Graph(id='feature-kde'), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='pressure-humidity-scatter', 
                                  figure=px.scatter(df, x='Pressure3pm', y='Humidity3pm', color='RainTomorrow',
                                                    opacity=0.4, title="Pressure vs Humidity Interaction")), width=12)
            ])
        ]),

        # Tab 3: Seasonality
        dbc.Tab(label="Seasonal Trends", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(df, x="month", color="RainTomorrow", barmode="group",
                                                      title="Rain Frequency by Month")), width=12)
            ], className="mt-4")
        ])
    ])
], fluid=True)

# --- Callbacks for Interactivity ---

@app.callback(
    [Output('feature-boxplot', 'figure'),
     Output('feature-kde', 'figure')],
    [Input('feature-dropdown', 'value')]
)
def update_feature_charts(selected_feature):
    # Box Plot
    box = px.box(df, x="RainTomorrow", y=selected_feature, color="RainTomorrow",
                 title=f"Distribution of {selected_feature}")
    
    # KDE (Approximated with Histogram/Violin for Plotly)
    kde = px.violin(df, y=selected_feature, x="RainTomorrow", color="RainTomorrow", 
                    box=True, points="all", title=f"Density Estimation of {selected_feature}")
    
    return box, kde

if __name__ == '__main__':
    app.run(debug=True)