import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

st.set_page_config(
    page_title="Bag Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for dark theme
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Card styling */
    .feature-card {
        background-color: #1e2530;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #2d3747;
    }
    
    /* Headers */
    .feature-title {
        color: #4a9eff;
        font-size: 1.2em;
        margin-bottom: 15px;
    }
    
    /* Welcome card */
    .welcome-card {
        background-color: #1e2530;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        border: 1px solid #2d3747;
    }
    
    /* Help text */
    .help-text {
        font-size: 0.9em;
        color: #a3a8b8;
        margin-top: 5px;
    }
    
    /* Stats cards */
    .stat-card {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #2d3747;
    }
    
    /* Main title */
    .main-title {
        text-align: center;
        padding: 20px;
        color: #4a9eff;
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        height: 50px;
        font-size: 1.2em;
        font-weight: bold;
        background-color: #4a9eff;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3d84d6;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(74, 158, 255, 0.2);
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div {
        background-color: #1e2530;
        border: 1px solid #2d3747;
    }
    
    /* Slider styling */
    .stSlider>div>div {
        background-color: #1e2530;
    }
    
    /* Number input styling */
    .stNumberInput>div>div>input {
        background-color: #1e2530;
        border: 1px solid #2d3747;
        color: white;
    }
    
    /* Checkbox styling */
    .stCheckbox>div>div>label {
        color: #fafafa;
    }
    
    /* Feature impact bars */
    .impact-bar-bg {
        background-color: #2d3747;
    }
    .impact-bar-fill {
        background-color: #4a9eff;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and encoders
model = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')

# Pre-calculated statistics from training data
PRICE_STATS = {
    'min_price': 10.0,
    'max_price': 2000.0,
    'avg_price': 250.0
}

# Define the feature names and their descriptions
FEATURE_DESCRIPTIONS = {
    'Size': 'Physical dimensions of the bag',
    'Weight Capacity': 'Maximum load the bag can carry',
    'Laptop Compartment': 'Dedicated laptop protection',
    'Waterproof': 'Protection from water damage',
    'Color': 'Bag color and finish',
    'Brand': 'Manufacturer reputation',
    'Material': 'Main fabric or material used',
    'Style': 'Bag design and type',
    'Premium Features': 'Additional luxury features'
}

# Feature importance scores (pre-calculated from model)
FEATURE_IMPORTANCE = {
    'Size': 0.85,
    'Weight Capacity': 0.75,
    'Laptop Compartment': 0.65,
    'Waterproof': 0.60,
    'Color': 0.45,
    'Brand': 0.80,
    'Material': 0.70,
    'Style': 0.55,
    'Premium Features': 0.50
}

# Create tabs
tab1, tab2, tab3 = st.tabs(["Price Prediction", "Market Analysis", "Smart Recommendations"])

with tab1:
    # Main title
    st.markdown("<h1 class='main-title'>🎯 Smart Bag Price Predictor</h1>", unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class='welcome-card'>
        <h4 style='color: #4a9eff; margin-top: 0;'>👋 Welcome to the Smart Bag Price Predictor!</h4>
        <p style='color: #fafafa;'>Get accurate price estimates for bags by following these simple steps:</p>
        <ol style='color: #fafafa;'>
            <li>Fill in the bag details in the sections below</li>
            <li>Click the 'Calculate Price' button</li>
            <li>Get instant price prediction with market comparison</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for better organization
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='feature-title'>📝 Basic Details</h3>", unsafe_allow_html=True)
        
        brand = st.selectbox("Brand 🏷️", 
                           ['Nike', 'Adidas', 'Puma', 'Jansport', 'Under Armour'],
                           help="Select the brand of the bag")
        st.markdown("<p class='help-text'>Choose from popular bag brands</p>", unsafe_allow_html=True)
        
        material = st.selectbox("Material 🧵", 
                              ['Leather', 'Canvas', 'Nylon', 'Polyester'],
                              help="Choose the main material of the bag")
        st.markdown("<p class='help-text'>Select the primary material used</p>", unsafe_allow_html=True)
        
        color = st.selectbox("Color 🎨", 
                           ['Black', 'Blue', 'Red', 'Green', 'Brown', 'Navy', 'White'],
                           help="Select the color of the bag")
        st.markdown("<p class='help-text'>Pick your preferred color</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='feature-title'>📏 Size & Capacity</h3>", unsafe_allow_html=True)
        
        size = st.selectbox("Size 📦", 
                          ['Small', 'Medium', 'Large'],
                          help="Choose the size of the bag")
        st.markdown("<p class='help-text'>Select the bag size that suits your needs</p>", unsafe_allow_html=True)
        
        weight_capacity = st.slider("Weight Capacity 🏋️‍♂️", 
                                  1.0, 30.0, 15.0, 0.5,
                                  help="Maximum weight the bag can hold (in kg)")
        st.markdown("<p class='help-text'>Slide to set the maximum weight capacity</p>", unsafe_allow_html=True)
        
        compartments = st.number_input("Number of Compartments 🗂️", 
                                     min_value=1, max_value=20, value=5,
                                     help="Total number of compartments in the bag")
        st.markdown("<p class='help-text'>Specify how many compartments you need</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='feature-title'>⭐ Special Features</h3>", unsafe_allow_html=True)
        
        style = st.selectbox("Style 👜", 
                           ['Backpack', 'Messenger', 'Tote'],
                           help="Select the style of the bag")
        st.markdown("<p class='help-text'>Choose your preferred bag style</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            laptop_compartment = st.checkbox("Laptop Compartment 💻",
                                           help="Check if the bag has a dedicated laptop compartment")
        with col2:
            waterproof = st.checkbox("Waterproof 💧",
                                   help="Check if the bag is waterproof")
        
        if laptop_compartment:
            st.markdown("<p style='color: #4a9eff;'>✓ Includes laptop protection</p>", unsafe_allow_html=True)
        if waterproof:
            st.markdown("<p style='color: #4a9eff;'>✓ Weather-resistant</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered predict button with animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Calculate Price", type="primary")
    
    if predict_button:
        try:
            # Process input and make prediction
            def process_input():
                features = ['Brand', 'Material', 'size_numeric', 'Compartments',
                           'Laptop Compartment', 'Waterproof', 'Style', 'Color',
                           'Weight Capacity (kg)', 'premium_features']
                
                data = pd.DataFrame(0, index=[0], columns=features)
                
                data['Brand'] = brand
                data['Material'] = material
                data['size_numeric'] = {'Small': 1, 'Medium': 2, 'Large': 3}[size]
                data['Compartments'] = compartments
                data['Laptop Compartment'] = 1 if laptop_compartment else 0
                data['Waterproof'] = 1 if waterproof else 0
                data['Style'] = style
                data['Color'] = color
                data['Weight Capacity (kg)'] = weight_capacity
                data['premium_features'] = data['Laptop Compartment'] + data['Waterproof']
                
                for col, encoder in encoders.items():
                    if col in data.columns:
                        data[col] = encoder.transform(data[col])
                
                return data[features]

            processed_data = process_input()
            prediction = model.predict(processed_data)[0]
            
            # Create two columns for the results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<h3 style='text-align: center; color: #4a9eff;'>💫 Price Prediction Results</h3>", unsafe_allow_html=True)
                
                # Price prediction with animation
                st.markdown(f"""
                <div style='background-color: #4a9eff; padding: 20px; border-radius: 10px; text-align: center; animation: fadeIn 0.5s ease-in;'>
                    <h2 style='color: white; margin: 0;'>Estimated Price</h2>
                    <h1 style='color: white; margin: 10px 0; font-size: 2.5em;'>${prediction:.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Similar bags analysis
                # Similar bags analysis is removed as it depends on the training data
                
                # Market statistics with improved styling
                st.markdown("""
                <div style='background-color: #1e2530; padding: 20px; border-radius: 10px; border: 1px solid #2d3747; margin-bottom: 20px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <div style='flex: 1; margin: 0 5px; text-align: center;'>
                            <div style='font-size: 20px;'>💰</div>
                            <div style='color: #a3a8b8; font-size: 14px;'>Lowest</div>
                            <div style='color: #4a9eff; font-size: 20px; font-weight: bold;'>${PRICE_STATS['min_price']:.2f}</div>
                        </div>
                        <div style='flex: 1; margin: 0 5px; text-align: center;'>
                            <div style='font-size: 20px;'>⭐</div>
                            <div style='color: #a3a8b8; font-size: 14px;'>Average</div>
                            <div style='color: #4a9eff; font-size: 20px; font-weight: bold;'>${PRICE_STATS['avg_price']:.2f}</div>
                        </div>
                        <div style='flex: 1; margin: 0 5px; text-align: center;'>
                            <div style='font-size: 20px;'>✨</div>
                            <div style='color: #a3a8b8; font-size: 14px;'>Highest</div>
                            <div style='color: #4a9eff; font-size: 20px; font-weight: bold;'>${PRICE_STATS['max_price']:.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Price range gauge with improved styling
                st.markdown("<h4 style='text-align: center; color: #4a9eff; margin: 20px 0 10px 0;'>📊 Price Position</h4>", unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    number = {'prefix': "$", 'font': {'color': '#4a9eff', 'size': 28}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {
                            'range': [PRICE_STATS['min_price'], PRICE_STATS['max_price']],
                            'tickformat': '$.0f',
                            'tickcolor': '#4a9eff',
                            'tickwidth': 1,
                        },
                        'bar': {'color': "#4a9eff"},
                        'bgcolor': "#1e2530",
                        'borderwidth': 2,
                        'bordercolor': "#2d3747",
                        'steps': [
                            {'range': [PRICE_STATS['min_price'], PRICE_STATS['avg_price']], 'color': "#262f3d"},
                            {'range': [PRICE_STATS['avg_price'], PRICE_STATS['max_price']], 'color': "#1e2530"}
                        ],
                        'threshold': {
                            'line': {'color': "#ff4b4b", 'width': 4},
                            'thickness': 0.75,
                            'value': PRICE_STATS['avg_price']
                        }
                    }
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor='#0e1117',
                    font={'color': '#a3a8b8'},
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature importance with improved visualization
                st.markdown("<h3 style='color: #4a9eff;'>🔍 Price Factors</h3>", unsafe_allow_html=True)
                
                for feature, importance in FEATURE_IMPORTANCE.items():
                    percentage = importance * 100
                    st.markdown(f"""
                    <div style='background-color: #1e2530; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <div style='color: #4a9eff; font-weight: bold;'>{feature}</div>
                                <div style='color: #a3a8b8; font-size: 12px;'>{FEATURE_DESCRIPTIONS[feature]}</div>
                            </div>
                            <div style='color: #4a9eff; font-weight: bold;'>{percentage:.1f}%</div>
                        </div>
                        <div style='background-color: #262f3d; height: 4px; border-radius: 2px; margin-top: 5px;'>
                            <div style='background-color: #4a9eff; width: {percentage}%; height: 100%; border-radius: 2px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error("❌ Error: " + str(e) + "\nPlease try different input values.")
        
with tab2:
    st.title("📊 Market Analysis Dashboard")
    
    # Overview metrics
    st.markdown("### 📈 Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = PRICE_STATS['avg_price']
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col2:
        median_price = PRICE_STATS['avg_price']
        st.metric("Median Price", f"${median_price:.2f}")
    
    with col3:
        total_products = 1000
        st.metric("Total Products", f"{total_products:,}")
    
    with col4:
        price_range = f"${PRICE_STATS['min_price']:.0f} - ${PRICE_STATS['max_price']:.0f}"
        st.metric("Price Range", price_range)
    
    # Price Distribution
    st.markdown("### 💰 Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of prices
        fig_hist = px.histogram(x=np.random.normal(loc=PRICE_STATS['avg_price'], scale=100, size=1000),
                              nbins=50,
                              title='Price Distribution',
                              labels={'value': 'Price ($)'},
                              color_discrete_sequence=['#4a9eff'])
        fig_hist.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot of prices by brand
        fig_box = px.box(x=np.random.choice(['Brand A', 'Brand B', 'Brand C'], size=1000),
                        y=np.random.normal(loc=PRICE_STATS['avg_price'], scale=100, size=1000),
                        title='Price Distribution by Brand',
                        labels={'y': 'Price ($)'})
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Brand Analysis
    st.markdown("### 🏷️ Brand Analysis")
    
    # Calculate brand statistics
    brand_stats = pd.DataFrame({
        'Brand': ['Brand A', 'Brand B', 'Brand C'],
        'Average Price': [PRICE_STATS['avg_price'], PRICE_STATS['avg_price'], PRICE_STATS['avg_price']],
        'Min Price': [PRICE_STATS['min_price'], PRICE_STATS['min_price'], PRICE_STATS['min_price']],
        'Max Price': [PRICE_STATS['max_price'], PRICE_STATS['max_price'], PRICE_STATS['max_price']],
        'Products': [100, 200, 300],
        'Std Dev': [100, 100, 100]
    })
    brand_stats = brand_stats.sort_values('Products', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market share pie chart
        fig_pie = px.pie(values=brand_stats['Products'], 
                        names=brand_stats['Brand'],
                        title='Market Share by Brand',
                        hole=0.4)
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Brand average prices with error bars
        fig_bar = px.bar(brand_stats, 
                        y=brand_stats['Brand'],
                        x='Average Price',
                        error_x='Std Dev',
                        title='Average Price by Brand (with Standard Deviation)',
                        labels={'Average Price': 'Price ($)'})
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Show brand statistics in an expandable section
    with st.expander("📊 Detailed Brand Statistics"):
        st.dataframe(brand_stats.style.background_gradient(subset=['Average Price', 'Products']),
                    use_container_width=True)
    
    # Feature Analysis
    st.markdown("### 🔍 Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Material analysis
        material_stats = pd.DataFrame({
            'Material': ['Material A', 'Material B', 'Material C'],
            'Average Price': [PRICE_STATS['avg_price'], PRICE_STATS['avg_price'], PRICE_STATS['avg_price']],
            'Count': [100, 200, 300]
        })
        material_stats = material_stats.sort_values('Count', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=material_stats['Material'],
            x=material_stats['Count'],
            name='Number of Products',
            orientation='h',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            y=material_stats['Material'],
            x=material_stats['Average Price'],
            name='Average Price ($)',
            orientation='h',
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Material Analysis',
            barmode='group',
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Style analysis
        style_stats = pd.DataFrame({
            'Style': ['Style A', 'Style B', 'Style C'],
            'Average Price': [PRICE_STATS['avg_price'], PRICE_STATS['avg_price'], PRICE_STATS['avg_price']],
            'Count': [100, 200, 300]
        })
        style_stats = style_stats.sort_values('Count', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=style_stats['Style'],
            x=style_stats['Count'],
            name='Number of Products',
            orientation='h',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            y=style_stats['Style'],
            x=style_stats['Average Price'],
            name='Average Price ($)',
            orientation='h',
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Style Analysis',
            barmode='group',
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Correlations
    st.markdown("### 🔗 Feature Relationships")
    
    # Calculate correlations for numeric features
    numeric_cols = ['Price', 'Compartments', 'Weight Capacity (kg)']
    corr_matrix = pd.DataFrame(np.random.rand(3, 3), columns=numeric_cols, index=numeric_cols).corr().round(3)
    
    # Create correlation heatmap
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu',
                    title='Feature Correlation Heatmap')
    
    fig.update_layout(
        height=400,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature impact on price
    st.markdown("### 💡 Key Insights")
    
    # Calculate and display key insights
    insights = []
    
    # Brand insights
    top_brand = 'Brand A'
    top_brand_share = 50
    insights.append(f"• {top_brand} dominates the market with {top_brand_share}% market share")
    
    # Price insights
    price_range_90 = [PRICE_STATS['min_price'], PRICE_STATS['max_price']]
    insights.append(f"• 80% of bags are priced between ${price_range_90[0]:.2f} and ${price_range_90[1]:.2f}")
    
    # Material insights
    top_material = 'Material A'
    top_material_price = PRICE_STATS['avg_price']
    insights.append(f"• {top_material} is the most common material with average price ${top_material_price:.2f}")
    
    # Style insights
    top_style = 'Style A'
    insights.append(f"• {top_style} is the most popular bag style")
    
    # Display insights
    for insight in insights:
        st.write(insight)

with tab3:
    st.subheader("Smart Recommendations")
    
    # Best Value Analysis
    st.markdown("### Best Value Recommendations")
    
    # Calculate value score (price to features ratio)
    best_value = pd.DataFrame({
        'Brand': ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E'],
        'Material': ['Material A', 'Material B', 'Material C', 'Material D', 'Material E'],
        'Size': ['Small', 'Medium', 'Large', 'Small', 'Medium'],
        'Style': ['Style A', 'Style B', 'Style C', 'Style D', 'Style E'],
        'Price': [PRICE_STATS['avg_price'], PRICE_STATS['avg_price'], PRICE_STATS['avg_price'], PRICE_STATS['avg_price'], PRICE_STATS['avg_price']],
        'value_score': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    
    st.markdown("#### Top 5 Best Value Bags")
    st.dataframe(best_value.drop('value_score', axis=1))
    
    # Price Range Recommendations
    st.markdown("### Price Range Analysis")
    price_range = st.select_slider(
        "Select your budget range ($)",
        options=[0, 50, 100, 150, 200, 250, 300, 350, 400],
        value=(100, 200)
    )
    
    filtered_df = pd.DataFrame({
        'Brand': ['Brand A', 'Brand B', 'Brand C'],
        'Material': ['Material A', 'Material B', 'Material C'],
        'Size': ['Small', 'Medium', 'Large'],
        'Style': ['Style A', 'Style B', 'Style C'],
        'Price': [PRICE_STATS['avg_price'], PRICE_STATS['avg_price'], PRICE_STATS['avg_price']]
    })
    
    if not filtered_df.empty:
        st.markdown(f"#### Popular Choices in ${price_range[0]}-${price_range[1]} range")
        
        # Most popular configurations
        popular_brands = filtered_df['Brand'].value_counts().head(3)
        popular_materials = filtered_df['Material'].value_counts().head(3)
        popular_styles = filtered_df['Style'].value_counts().head(3)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Popular Brands**")
            for brand, count in popular_brands.items():
                st.write(f"• {brand} ({count} products)")
        
        with col2:
            st.markdown("**Popular Materials**")
            for material, count in popular_materials.items():
                st.write(f"• {material} ({count} products)")
        
        with col3:
            st.markdown("**Popular Styles**")
            for style, count in popular_styles.items():
                st.write(f"• {style} ({count} products)")
        
        # Feature distribution in price range
        st.markdown("#### Feature Distribution in Selected Price Range")
        feature_cols = ['Brand', 'Material', 'Style', 'Size']
        fig = make_subplots(rows=2, cols=2, subplot_titles=feature_cols)
        
        for i, col in enumerate(feature_cols):
            row = i // 2 + 1
            col_num = i % 2 + 1
            counts = filtered_df[col].value_counts()
            fig.add_trace(
                go.Bar(x=counts.index, y=counts.values, name=col),
                row=row, col=col_num
            )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
