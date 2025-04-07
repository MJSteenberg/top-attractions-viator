import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Viator Attractions Analysis",
    page_icon="🌏",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.title("🌏 Viator Attractions Explorer")
st.markdown("""
Analyze worldwide tourist attractions based on popularity, tour offerings, and venue type.
Use this dashboard to discover travel insights across different regions.
""")

# Function to load data
@st.cache_data
def load_data():
    # Try to find the CSV file
    possible_paths = [
        "attraction_details.csv",
        "data/processed/attraction_details.csv",
        "attractions/viator-attractions-scraper/data/processed/attraction_details.csv",
        "../attractions/viator-attractions-scraper/data/processed/attraction_details.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # If file not found, let user upload
    st.warning("Could not find attraction data file. Please upload the CSV file.")
    uploaded_file = st.file_uploader("Upload attraction_details.csv", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        st.stop()

# Load and prepare data
df = load_data()

# Clean and prepare the data
@st.cache_data
def prepare_data(df):
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Convert tour_count to numeric
    if 'tour_count' in data.columns:
        data['tour_count'] = pd.to_numeric(data['tour_count'], errors='coerce')
    
    # Convert rating to numeric
    if 'rating' in data.columns:
        data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    
    # Convert review_count to numeric
    if 'review_count' in data.columns:
        data['review_count'] = pd.to_numeric(data['review_count'], errors='coerce')
        
    # Format region names
    if 'region' in data.columns:
        data['region'] = data['region'].str.replace('_', ' ')
    
    # Make sure venue_type exists and has proper values
    if 'venue_type' not in data.columns:
        data['venue_type'] = 'unknown'
    else:
        data['venue_type'] = data['venue_type'].fillna('unknown')
    
    # Fix any missing price data
    if 'price' in data.columns:
        data['price'] = pd.to_numeric(data['price'], errors='coerce')
    
    return data

# Prepare the data
data = prepare_data(df)

# Get unique attractions
@st.cache_data
def get_attractions_summary(data):
    # Group by attraction name, region, and venue_type and aggregate
    attractions_summary = data.groupby(['attraction_name', 'region', 'venue_type', 'tour_count']).agg({
        'rating': 'mean',
        'review_count': 'sum',
    }).reset_index()
    
    # Rename the columns for display
    attractions_summary = attractions_summary.rename(columns={
        'tour_count': 'attraction_tour_count',
        'review_count': 'tour_review_count'
    })
    
    # Calculate average price per attraction if available
    if 'price' in data.columns:
        price_data = data.groupby(['attraction_name']).agg({
            'price': 'mean'
        }).reset_index()
        
        # Merge price data
        attractions_summary = pd.merge(
            attractions_summary,
            price_data,
            on='attraction_name',
            how='left'
        )
    
    return attractions_summary

# Get attraction summary
attractions_summary = get_attractions_summary(data)

# Geocode attractions for mapping
@st.cache_data
def geocode_attractions(attractions_df):
    # Create a geocoder
    geolocator = Nominatim(user_agent="viator_attractions_explorer")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    # Create dataframe for geocoded data
    geo_data = attractions_df.copy()
    geo_data['latitude'] = np.nan
    geo_data['longitude'] = np.nan
    
    # This could take a long time for many attractions, so let's use a progress bar
    total_attractions = len(geo_data)
    
    # Check if we already have a cache file
    cache_file = Path("geocode_cache.csv")
    cache = {}
    
    if cache_file.exists():
        cache_df = pd.read_csv(cache_file)
        cache = dict(zip(cache_df['query'], zip(cache_df['latitude'], cache_df['longitude'])))
    
    for i, row in enumerate(geo_data.iterrows()):
        idx, attraction = row
        query = f"{attraction['attraction_name']}, {attraction['region']}"
        
        # Check cache first
        if query in cache:
            geo_data.at[idx, 'latitude'] = cache[query][0]
            geo_data.at[idx, 'longitude'] = cache[query][1]
            continue
        
        # If not in cache, geocode
        try:
            location = geocode(query)
            if location:
                geo_data.at[idx, 'latitude'] = location.latitude
                geo_data.at[idx, 'longitude'] = location.longitude
                
                # Add to cache
                cache[query] = (location.latitude, location.longitude)
            else:
                # Try with just the attraction name
                location = geocode(attraction['attraction_name'])
                if location:
                    geo_data.at[idx, 'latitude'] = location.latitude
                    geo_data.at[idx, 'longitude'] = location.longitude
                    
                    # Add to cache
                    cache[query] = (location.latitude, location.longitude)
        except Exception as e:
            pass
    
    # Save updated cache
    cache_df = pd.DataFrame({
        'query': list(cache.keys()),
        'latitude': [lat for lat, lon in cache.values()],
        'longitude': [lon for lat, lon in cache.values()]
    })
    cache_df.to_csv(cache_file, index=False)
    
    # Filter out any rows with missing coordinates
    geo_data = geo_data.dropna(subset=['latitude', 'longitude'])
    
    return geo_data

# Since geocoding is expensive and time-consuming, we'll add a button to trigger it
if 'geocoded_data' not in st.session_state:
    st.session_state.geocoded_data = None

# Add filters in the sidebar
st.sidebar.header("Filters")

# Region filter - change to select all regions by default
available_regions = sorted(data['region'].unique())
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=available_regions,
    default=available_regions  # Changed from available_regions[:3] to select all regions
)

# Venue type filter - change to select only "indoor" by default
venue_types = sorted(data['venue_type'].unique())
selected_venue_types = st.sidebar.multiselect(
    "Venue Type",
    options=venue_types,
    default=["indoor"] if "indoor" in venue_types else []  # Default to just "indoor" instead of all types
)

# Apply filters to data
if selected_regions and selected_venue_types:
    filtered_data = data[data['region'].isin(selected_regions) & 
                         data['venue_type'].isin(selected_venue_types)]
    
    filtered_attractions = attractions_summary[
        attractions_summary['region'].isin(selected_regions) & 
        attractions_summary['venue_type'].isin(selected_venue_types)
    ]
else:
    filtered_data = data
    filtered_attractions = attractions_summary

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Top Attractions", 
    "🗺️ Map View", 
    "📈 Region Analysis",
    "🔍 Detailed Data"
])

with tab1:
    st.header("Top Tourist Attractions")
    
    # Radio button to select ranking method
    ranking_method = st.radio(
        "Rank by:",
        options=["Tour Count", "Review Count", "Average Rating"],
        horizontal=True
    )
    
    # Display top attractions based on selected method
    if ranking_method == "Tour Count":
        top_attractions = filtered_attractions.sort_values('attraction_tour_count', ascending=False).head(50)
        metric_col = 'attraction_tour_count'
        metric_name = 'Number of Tours'
    elif ranking_method == "Review Count":
        top_attractions = filtered_attractions.sort_values('tour_review_count', ascending=False).head(50)
        metric_col = 'tour_review_count'
        metric_name = 'Total Reviews'
    else:  # Average Rating
        top_attractions = filtered_attractions.sort_values('rating', ascending=False).head(50)
        metric_col = 'rating'
        metric_name = 'Average Rating'
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create bar chart
        fig = px.bar(
            top_attractions,
            y='attraction_name',
            x=metric_col,
            color='region',
            labels={
                'attraction_name': 'Attraction',
                metric_col: metric_name,
                'region': 'Region'
            },
            title=f"Top 50 Attractions by {metric_name}",
            orientation='h',
            height=1000
        )
        
        # Update layout
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            hovermode='closest',
            xaxis_title=metric_name,
            yaxis_title=None
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display top attractions by region
        st.subheader("Top Attraction in Each Region")
        
        for region in selected_regions:
            region_attractions = filtered_attractions[filtered_attractions['region'] == region]
            if not region_attractions.empty:
                top_attraction = region_attractions.sort_values(metric_col, ascending=False).iloc[0]
                
                # Create a card-like display
                st.markdown(f"""
                <div style="
                    background-color: white;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin-top: 0;">{region}</h4>
                    <p style="font-weight: bold; margin-bottom: 5px;">{top_attraction['attraction_name']}</p>
                    <p style="margin: 0;">Tours: {int(top_attraction['attraction_tour_count']) if not pd.isna(top_attraction['attraction_tour_count']) else 'N/A'}</p>
                    <p style="margin: 0;">Rating: {top_attraction['rating']:.1f}/5.0</p>
                    <p style="margin: 0;">Reviews: {int(top_attraction['tour_review_count']) if not pd.isna(top_attraction['tour_review_count']) else 'N/A'}</p>
                    <p style="margin: 0;">Type: {top_attraction['venue_type'].capitalize()}</p>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.header("Attractions Map")
    
    # Button to trigger geocoding
    if st.session_state.geocoded_data is None:
        if st.button("Generate Map (May take a while for first run)"):
            with st.spinner("Geocoding attractions... This may take a while for the first run."):
                st.session_state.geocoded_data = geocode_attractions(filtered_attractions)
    
    # If we have geocoded data, display the map
    if st.session_state.geocoded_data is not None:
        # Filter the geocoded data based on selections
        geo_data = st.session_state.geocoded_data[
            st.session_state.geocoded_data['region'].isin(selected_regions) & 
            st.session_state.geocoded_data['venue_type'].isin(selected_venue_types)
        ]
        
        if not geo_data.empty:
            # Determine the metric for sizing points
            map_metric = st.radio(
                "Size points by:",
                options=["Tour Count", "Review Count"],
                horizontal=True
            )
            
            # Create scatter plot on a map
            if map_metric == "Tour Count":
                size_col = 'attraction_tour_count'
            else:  # Review Count
                size_col = 'tour_review_count'
            
            # Normalize sizes for better visualization
            geo_data['point_size'] = np.log1p(geo_data[size_col]) * 5  # log scale to prevent huge points
            
            # Create map
            fig = px.scatter_mapbox(
                geo_data,
                lat="latitude",
                lon="longitude",
                hover_name="attraction_name",
                hover_data={
                    "attraction_tour_count": True,
                    "rating": ":.1f",
                    "tour_review_count": True,
                    "venue_type": True,
                    "latitude": False,
                    "longitude": False,
                    "point_size": False
                },
                color="venue_type",
                size="point_size",
                size_max=20,
                zoom=1.5,
                height=600,
                color_discrete_map={
                    "indoor": "#1E88E5",
                    "outdoor": "#43A047",
                    "mixed": "#FBC02D",
                    "unknown": "#757575"
                }
            )
            
            # Update map layout
            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"r":0,"t":0,"l":0,"b":0},
                legend_title_text="Venue Type"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add note about the map
            st.info(
                "The map shows attractions based on geocoded locations. Point size represents " +
                (f"the number of tours offered" if map_metric == "Tour Count" else "the number of reviews") +
                ". Hover over points for more details."
            )
        else:
            st.warning("No attractions with geographic coordinates match your current filters.")
    else:
        st.info("Click the 'Generate Map' button to geocode attractions and display them on a map.")

with tab3:
    st.header("Region Analysis")
    
    # Show summary statistics by region
    region_stats = filtered_attractions.groupby('region').agg({
        'attraction_name': 'count',
        'attraction_tour_count': 'mean',
        'rating': 'mean',
        'tour_review_count': 'mean',
        'venue_type': lambda x: x.value_counts().index[0]  # Most common venue type
    }).reset_index()
    
    region_stats.columns = ['Region', 'Attraction Count', 'Avg Tours per Attraction', 
                           'Avg Rating', 'Avg Reviews per Attraction', 'Most Common Venue Type']
    
    # Format columns
    region_stats['Avg Tours per Attraction'] = region_stats['Avg Tours per Attraction'].round(1)
    region_stats['Avg Rating'] = region_stats['Avg Rating'].round(2)
    region_stats['Avg Reviews per Attraction'] = region_stats['Avg Reviews per Attraction'].round(1)
    
    # Sort by attraction count
    region_stats = region_stats.sort_values('Attraction Count', ascending=False)
    
    # Display the table
    st.dataframe(region_stats, use_container_width=True)
    
    # Show venue type distribution by region
    st.subheader("Venue Type Distribution by Region")
    
    # Calculate venue type counts by region
    venue_counts = filtered_attractions.groupby(['region', 'venue_type']).size().reset_index(name='count')
    
    # Create stacked bar chart
    fig = px.bar(
        venue_counts,
        x='region',
        y='count',
        color='venue_type',
        labels={'region': 'Region', 'count': 'Number of Attractions', 'venue_type': 'Venue Type'},
        title="Number of Indoor vs Outdoor Attractions by Region",
        color_discrete_map={
            "indoor": "#1E88E5",
            "outdoor": "#43A047",
            "mixed": "#FBC02D",
            "unknown": "#757575"
        }
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Number of Attractions",
        legend_title="Venue Type",
        xaxis={'categoryorder': 'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show average tour count and review count by venue type
    st.subheader("Comparison by Venue Type")
    
    # Calculate averages by venue type
    venue_averages = filtered_attractions.groupby('venue_type').agg({
        'attraction_tour_count': 'mean',
        'tour_review_count': 'mean',
        'rating': 'mean'
    }).reset_index()
    
    # Create 3 columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Tour count by venue type
        fig = px.bar(
            venue_averages,
            x='venue_type',
            y='attraction_tour_count',
            color='venue_type',
            labels={'venue_type': 'Venue Type', 'attraction_tour_count': 'Average Tours'},
            title="Avg Tours by Venue Type",
            color_discrete_map={
                "indoor": "#1E88E5",
                "outdoor": "#43A047",
                "mixed": "#FBC02D",
                "unknown": "#757575"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Review count by venue type
        fig = px.bar(
            venue_averages,
            x='venue_type',
            y='tour_review_count',
            color='venue_type',
            labels={'venue_type': 'Venue Type', 'tour_review_count': 'Average Reviews'},
            title="Avg Reviews by Venue Type",
            color_discrete_map={
                "indoor": "#1E88E5",
                "outdoor": "#43A047",
                "mixed": "#FBC02D",
                "unknown": "#757575"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Rating by venue type
        fig = px.bar(
            venue_averages,
            x='venue_type',
            y='rating',
            color='venue_type',
            labels={'venue_type': 'Venue Type', 'rating': 'Average Rating'},
            title="Avg Rating by Venue Type",
            color_discrete_map={
                "indoor": "#1E88E5",
                "outdoor": "#43A047",
                "mixed": "#FBC02D",
                "unknown": "#757575"
            }
        )
        # Set y-axis to start from a reasonable minimum for ratings
        fig.update_layout(yaxis_range=[min(3.5, venue_averages['rating'].min() - 0.2), 5.0])
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Detailed Data Explorer")
    
    # Text search filter
    search_term = st.text_input("Search for attractions:")
    
    # Apply text search if provided
    if search_term:
        search_results = filtered_data[filtered_data['attraction_name'].str.contains(search_term, case=False)]
    else:
        search_results = filtered_data
    
    # Reorder columns to put attraction_name first and title second
    if not search_results.empty:
        # Get list of all columns
        all_cols = list(search_results.columns)
        
        # Remove attraction_name and title from the list (if they exist)
        cols_to_reorder = [col for col in all_cols if col not in ['attraction_name', 'title']]
        
        # Create new column order with attraction_name first and title second
        new_col_order = ['attraction_name', 'title'] + cols_to_reorder
        
        # Make sure all columns in new_col_order exist in the dataframe
        new_col_order = [col for col in new_col_order if col in all_cols]
        
        # Reorder the columns
        search_results = search_results[new_col_order]
        
        # Sort by review_count in descending order by default
        if 'review_count' in search_results.columns:
            search_results = search_results.sort_values('review_count', ascending=False)
    
    # Show the filtered data
    st.dataframe(search_results, use_container_width=True)
    
    # Option to download filtered data
    if not search_results.empty:
        csv = search_results.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_attractions.csv",
            mime="text/csv"
        )

# Footer
st.markdown("""
---
### About this Dashboard

This dashboard visualizes global tourist attraction data collected from Viator. The data includes:
- Attraction information (name, region, venue type)
- Popularity metrics (tour count, review count, ratings)

Use the filters in the sidebar to focus on specific regions or venue types.
""")