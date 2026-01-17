import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the data
game_final = pd.read_csv('game_final.csv')

# Prepare clustering data
features_of_interest = ['review_score', 'main', 'rating', 'Total_Global_Sales']
cluster_data = game_final[features_of_interest + ['name']].dropna()
cluster_data = cluster_data[cluster_data['main'] < 200]  # Remove outliers

X = cluster_data[features_of_interest].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
cluster_data['Cluster'] = cluster_labels

# Assign archetypes
def assign_archetype(cluster_id):
    subset = cluster_data[cluster_data['Cluster'] == cluster_id]
    avg_sales = subset['Total_Global_Sales'].mean()
    avg_review = subset['review_score'].mean()
    avg_rating = subset['rating'].mean()
    
    if avg_sales > 2.5:
        return "🏆 Blockbusters"
    elif avg_review > 70 and avg_rating > 70:
        return "💎 Hidden Gems"
    else:
        return "🎮 Casual/Mainstream"

cluster_data['Archetype'] = cluster_data['Cluster'].apply(assign_archetype)

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
cluster_data['PC1'] = X_pca[:, 0]
cluster_data['PC2'] = X_pca[:, 1]

# Prepare regional sales data
genre_cols = [col for col in game_final.columns if col.startswith('Genre_')]
regional_sales = {}
for genre_col in genre_cols:
    genre_games = game_final[game_final[genre_col] == 1]
    genre_name = genre_col.replace('Genre_', '')
    regional_sales[genre_name] = {
        'North America': genre_games['Total_NA_Sales'].sum(),
        'Europe': genre_games['Total_EU_Sales'].sum(),
        'Japan': genre_games['Total_JP_Sales'].sum()
    }

regional_df = pd.DataFrame(regional_sales).T.reset_index()
regional_df.columns = ['Genre', 'North America', 'Europe', 'Japan']
regional_df_melted = regional_df.melt(id_vars='Genre', var_name='Region', value_name='Sales')

# Prepare genre evolution data
genre_evolution = game_final.melt(
    id_vars=['Year', 'Total_Global_Sales'],
    value_vars=genre_cols,
    var_name='Genre',
    value_name='Is_Genre'
)
genre_evolution = genre_evolution[genre_evolution['Is_Genre'] == 1]
genre_evolution['Genre'] = genre_evolution['Genre'].str.replace('Genre_', '')
yearly_genre_sales = genre_evolution.groupby(['Year', 'Genre'])['Total_Global_Sales'].sum().reset_index()

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server for deployment

# Define colors
cluster_colors = {
    '🏆 Blockbusters': '#E91E63',
    '💎 Hidden Gems': '#00E676',
    '🎮 Casual/Mainstream': '#2196F3'
}

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Video Game Market Analysis Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Interactive exploration of video game sales, quality metrics, and market segments",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
    
    # Filters
    html.Div([
        html.Div([
            html.Label("Select Region:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='region-filter',
                options=[
                    {'label': 'All Regions', 'value': 'All'},
                    {'label': 'North America', 'value': 'North America'},
                    {'label': 'Europe', 'value': 'Europe'},
                    {'label': 'Japan', 'value': 'Japan'}
                ],
                value='All',
                style={'width': '200px'}
            )
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        html.Div([
            html.Label("Select Cluster:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='cluster-filter',
                options=[
                    {'label': 'All Clusters', 'value': 'All'},
                    {'label': '🏆 Blockbusters', 'value': '🏆 Blockbusters'},
                    {'label': '💎 Hidden Gems', 'value': '💎 Hidden Gems'},
                    {'label': '🎮 Casual/Mainstream', 'value': '🎮 Casual/Mainstream'}
                ],
                value='All',
                style={'width': '250px'}
            )
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        html.Div([
            html.Label("Sales Range (Millions):", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='sales-slider',
                min=0,
                max=50,
                step=1,
                value=[0, 50],
                marks={0: '0', 10: '10M', 20: '20M', 30: '30M', 40: '40M', 50: '50M+'},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={'display': 'inline-block', 'width': '400px'})
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'marginBottom': '20px', 'textAlign': 'center'}),
    
    # First row: Clustering + Regional Sales
    html.Div([
        html.Div([
            dcc.Graph(id='cluster-scatter', style={'height': '450px'})
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign': 'top'}),
        
        html.Div([
            dcc.Graph(id='regional-bar', style={'height': '450px'})
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign': 'top'})
    ], style={'textAlign': 'center', 'margin': '0 auto'}),
    
    # Second row: Review vs Sales + Genre Evolution
    html.Div([
        html.Div([
            dcc.Graph(id='review-scatter', style={'height': '450px'})
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign': 'top'}),
        
        html.Div([
            dcc.Graph(id='genre-evolution', style={'height': '450px'})
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign': 'top'})
    ], style={'textAlign': 'center', 'margin': '0 auto'})
], style={'backgroundColor': '#f8f9fa', 'fontFamily': 'Arial, sans-serif', 'margin': '0', 'padding': '0'})

# Callbacks for interactivity
@app.callback(
    [Output('cluster-scatter', 'figure'),
     Output('review-scatter', 'figure')],
    [Input('cluster-filter', 'value'),
     Input('sales-slider', 'value')]
)
def update_scatter_plots(cluster_filter, sales_range):
    # Filter data
    filtered_data = cluster_data.copy()
    
    if cluster_filter != 'All':
        filtered_data = filtered_data[filtered_data['Archetype'] == cluster_filter]
    
    filtered_data = filtered_data[
        (filtered_data['Total_Global_Sales'] >= sales_range[0]) &
        (filtered_data['Total_Global_Sales'] <= sales_range[1])
    ]
    
    # Sample data if too many points (to avoid WebGL issues)
    max_points = 1000
    if len(filtered_data) > max_points:
        filtered_data = filtered_data.sample(n=max_points, random_state=42)
    
    # Cluster scatter plot
    fig1 = px.scatter(
        filtered_data,
        x='PC1',
        y='PC2',
        color='Archetype',
        color_discrete_map=cluster_colors,
        hover_data={'name': True, 'Total_Global_Sales': ':.2f', 'review_score': ':.0f', 
                    'PC1': False, 'PC2': False},
        labels={'PC1': f'First Principal Component ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                'PC2': f'Second Principal Component ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'},
        title=f'Market Segmentation (K-Means Clustering) - Showing {len(filtered_data)} games'
    )
    fig1.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='black')))
    fig1.update_layout(
        template='plotly_white',
        hovermode='closest',
        legend=dict(title='Market Segment', orientation='v', x=1.02, y=1)
    )
    
    # Review vs Sales scatter
    fig2 = px.scatter(
        filtered_data,
        x='review_score',
        y='Total_Global_Sales',
        color='Archetype',
        color_discrete_map=cluster_colors,
        hover_data={'name': True, 'rating': ':.0f'},
        labels={'review_score': 'Review Score', 'Total_Global_Sales': 'Global Sales (Millions)'},
        title=f'Review Score vs Global Sales - Showing {len(filtered_data)} games'
    )
    fig2.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='black')))
    fig2.update_layout(
        template='plotly_white',
        hovermode='closest',
        legend=dict(title='Market Segment', orientation='v', x=1.02, y=1)
    )
    
    return fig1, fig2

@app.callback(
    Output('regional-bar', 'figure'),
    [Input('region-filter', 'value')]
)
def update_regional_bar(region_filter):
    if region_filter == 'All':
        data_to_plot = regional_df_melted
    else:
        data_to_plot = regional_df_melted[regional_df_melted['Region'] == region_filter]
    
    fig = px.bar(
        data_to_plot,
        x='Genre',
        y='Sales',
        color='Region',
        barmode='group',
        title='Genre Popularity by Region',
        labels={'Sales': 'Sales (Millions)'},
        color_discrete_map={
            'North America': '#3498db',
            'Europe': '#e74c3c',
            'Japan': '#f39c12'
        }
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=-45,
        legend=dict(title='Region', orientation='v', x=1.02, y=1)
    )
    
    return fig

@app.callback(
    Output('genre-evolution', 'figure'),
    [Input('cluster-filter', 'value')]
)
def update_genre_evolution(cluster_filter):
    # Get top 5 genres
    top_genres = yearly_genre_sales.groupby('Genre')['Total_Global_Sales'].sum().nlargest(5).index
    filtered_evolution = yearly_genre_sales[yearly_genre_sales['Genre'].isin(top_genres)].copy()
    
    # Remove any invalid data
    filtered_evolution = filtered_evolution.dropna()
    
    # Create figure
    fig = go.Figure()
    
    for genre in top_genres:
        genre_data = filtered_evolution[filtered_evolution['Genre'] == genre]
        fig.add_trace(go.Scatter(
            x=genre_data['Year'],
            y=genre_data['Total_Global_Sales'],
            mode='lines+markers',
            name=genre,
            line=dict(width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='Evolution of Top 5 Genres Over Time',
        xaxis_title='Year',
        yaxis_title='Total Sales (Millions)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(title='Genre', orientation='v', x=1.02, y=1)
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)
