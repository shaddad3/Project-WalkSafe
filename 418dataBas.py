import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen

def load_and_clean_data(csv_path):
    """Load and clean the crash data"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Clean coordinates
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    df = df[
        (df['LATITUDE'].between(41.6, 42.1)) & 
        (df['LONGITUDE'].between(-88.0, -87.5))
    ]
    return df[~(df['LATITUDE'].isin([0, None])) & ~(df['LONGITUDE'].isin([0, None]))]

def create_interactive_map(df, output_file='chicago_pedestrian_crashes.html'):
    """Create an interactive map of pedestrian crashes with enhanced features"""
    # Filter pedestrian crashes
    pedestrian = df[df['FIRST_CRASH_TYPE'] == 'PEDESTRIAN'].copy()
    
    # Parse datetime with multiple format attempts
    datetime_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %I:%M:%S %p',
        '%Y-%m-%d',
        '%m/%d/%Y %H:%M'
    ]
    
    pedestrian['CRASH_DATETIME'] = pd.NaT
    for fmt in datetime_formats:
        mask = pedestrian['CRASH_DATETIME'].isna()
        pedestrian.loc[mask, 'CRASH_DATETIME'] = pd.to_datetime(
            pedestrian.loc[mask, 'CRASH_DATE'], 
            format=fmt, 
            errors='coerce'
        )
    
    # Categorize crash severity
    conditions = [
        (pedestrian['MOST_SEVERE_INJURY'] == 'FATAL'),
        (pedestrian['MOST_SEVERE_INJURY'] == 'INCAPACITATING INJURY'),
        (pedestrian['INJURIES_TOTAL'] > 0),
        (pedestrian['DAMAGE'] == 'OVER $1,500')
    ]
    choices = ['Fatal', 'Severe', 'Injury', 'Property Damage']
    pedestrian['SEVERITY_TIER'] = np.select(conditions, choices, default='Minor')
    
    # Add time information
    pedestrian['DAY_OF_WEEK'] = pedestrian['CRASH_DATETIME'].dt.day_name()
    pedestrian['HOUR'] = pedestrian['CRASH_DATETIME'].dt.hour
    pedestrian['TIME_OF_DAY'] = np.where(
        pedestrian['HOUR'].isna(),
        'Unknown',
        pd.cut(
            pedestrian['HOUR'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night (12am-6am)', 'Morning (6am-12pm)', 
                   'Afternoon (12pm-6pm)', 'Evening (6pm-12am)']
        )
    )
    
    # Create full street address for display
    pedestrian['FULL_ADDRESS'] = (
        pedestrian['STREET_NO'].astype(str) + ' ' +
        pedestrian['STREET_DIRECTION'] + ' ' +
        pedestrian['STREET_NAME']
    )
    
    # Create the base map
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, 
                  tiles='CartoDB positron', control_scale=True)
    
    # Add alternative map tiles with proper attribution
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer(
        tiles='Stamen Toner',
        attr='Map tiles by <a href="https://stamen.com">Stamen Design</a>, under <a href="https://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="https://openstreetmap.org">OpenStreetMap</a>, under <a href="https://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)
    folium.TileLayer(
        tiles='CartoDB dark_matter',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    ).add_to(m)
    
    # Configure marker styling
    severity_style = {
        'Fatal': {'color': '#d62728', 'size': 8},
        'Severe': {'color': '#ff7f0e', 'size': 6},
        'Injury': {'color': '#1f77b4', 'size': 5},
        'Property Damage': {'color': '#2ca02c', 'size': 4},
        'Minor': {'color': '#7f7f7f', 'size': 3}
    }
    
    # Create marker clusters for each severity type
    severity_clusters = {
        'Fatal': MarkerCluster(name='Fatal Accidents').add_to(m),
        'Severe': MarkerCluster(name='Severe Accidents').add_to(m),
        'Injury': MarkerCluster(name='Injury Accidents').add_to(m),
        'Property Damage': MarkerCluster(name='Property Damage').add_to(m),
        'Minor': MarkerCluster(name='Minor Accidents').add_to(m)
    }
    
    # Add markers to their respective clusters
    for _, row in pedestrian.iterrows():
        date_str = row['CRASH_DATETIME'].strftime('%Y-%m-%d') if pd.notnull(row['CRASH_DATETIME']) else 'N/A'
        day_str = row['DAY_OF_WEEK'] if pd.notnull(row['DAY_OF_WEEK']) else 'Unknown'
        time_str = row['TIME_OF_DAY'] if pd.notnull(row['TIME_OF_DAY']) else 'Unknown'
        
        popup_html = f"""
        <b>Severity:</b> {row['SEVERITY_TIER']}<br>
        <b>Date:</b> {date_str}<br>
        <b>Time:</b> {time_str} ({day_str})<br>
        <b>Injuries:</b> {row.get('INJURIES_TOTAL', 0)}<br>
        <b>Speed Limit:</b> {row.get('POSTED_SPEED_LIMIT', 'N/A')} mph<br>
        <b>Weather:</b> {row.get('WEATHER_CONDITION', 'N/A')}<br>
        <b>Address:</b> {row.get('FULL_ADDRESS', 'N/A')}
        """
        
        marker = folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=severity_style[row['SEVERITY_TIER']]['size'],
            color=severity_style[row['SEVERITY_TIER']]['color'],
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['SEVERITY_TIER']} accident at {time_str}"
        )
        severity_clusters[row['SEVERITY_TIER']].add_child(marker)
    
    # Add map controls
    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    MiniMap(tile_layer='CartoDB positron', position='bottomright').add_to(m)
    Fullscreen(position='topleft').add_to(m)
    
    # Severity definitions for popups
    severity_definitions = {
        'Fatal': 'A crash where at least one person died as a result of the accident (MOST_SEVERE_INJURY = "FATAL")',
        'Severe': 'Crashes involving incapacitating injuries that prevent normal activities (MOST_SEVERE_INJURY = "INCAPACITATING INJURY")',
        'Injury': 'Crashes with non-incapacitating but reportable injuries (INJURIES_TOTAL > 0)',
        'Property Damage': 'Crashes with only vehicle/property damage (DAMAGE = "OVER $1,500") and no injuries',
        'Minor': 'All other reportable crashes that don\'t meet the above criteria (default classification)'
    }
    
    # Add interactive legend with clickable items
    legend_html = f'''
    <div id="legend" style="position: fixed; bottom: 50px; left: 50px; width: 220px; 
                background: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey; z-index: 9999; font-size: 14px;
                box-shadow: 3px 3px 5px grey;">
        <h4 style="margin:5px 0">Accident Severity</h4>
        <div style="display:flex;align-items:center;cursor:pointer;" onclick="showDefinition('Fatal')">
            <div style="background:#d62728;width:12px;height:12px;border-radius:50%;margin-right:5px"></div>
            <span>Fatal</span>
        </div>
        <div style="display:flex;align-items:center;cursor:pointer;" onclick="showDefinition('Severe')">
            <div style="background:#ff7f0e;width:12px;height:12px;border-radius:50%;margin-right:5px"></div>
            <span>Severe</span>
        </div>
        <div style="display:flex;align-items:center;cursor:pointer;" onclick="showDefinition('Injury')">
            <div style="background:#1f77b4;width:12px;height:12px;border-radius:50%;margin-right:5px"></div>
            <span>Injury</span>
        </div>
        <div style="display:flex;align-items:center;cursor:pointer;" onclick="showDefinition('Property Damage')">
            <div style="background:#2ca02c;width:12px;height:12px;border-radius:50%;margin-right:5px"></div>
            <span>Property Damage</span>
        </div>
        <div style="display:flex;align-items:center;cursor:pointer;" onclick="showDefinition('Minor')">
            <div style="background:#7f7f7f;width:12px;height:12px;border-radius:50%;margin-right:5px"></div>
            <span>Minor</span>
        </div>
    </div>
    
    <div id="definition-popup" style="display:none; position: fixed; bottom: 50px; left: 280px; width: 300px; 
                background: white; padding: 15px; border-radius: 5px;
                border: 2px solid grey; z-index: 9999; font-size: 14px;
                box-shadow: 3px 3px 5px grey;">
        <h4 id="popup-title" style="margin-top:0;"></h4>
        <p id="popup-content"></p>
        <button onclick="document.getElementById('definition-popup').style.display='none'" 
                style="float:right; padding: 3px 8px; margin-top:5px;">Close</button>
    </div>
    
    <script>
    const definitions = {{
        'Fatal': `{severity_definitions['Fatal']}`,
        'Severe': `{severity_definitions['Severe']}`,
        'Injury': `{severity_definitions['Injury']}`,
        'Property Damage': `{severity_definitions['Property Damage']}`,
        'Minor': `{severity_definitions['Minor']}`
    }};
    
    function showDefinition(severity) {{
        const popup = document.getElementById('definition-popup');
        document.getElementById('popup-title').textContent = severity + ' Accidents';
        document.getElementById('popup-content').textContent = definitions[severity];
        popup.style.display = 'block';
    }}
    </script>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save and return the map
    m.save(output_file)
    print(f"Interactive map saved to {output_file}")
    return m

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_and_clean_data('/Users/alaa/Documents/418Analysis/Traffic_Crashes_-_Crashes_20250331.csv')
    print(f"Loaded {len(df)} crash records")
    
    # Create and save the interactive map
    create_interactive_map(df)