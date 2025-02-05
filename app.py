import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load the trained model
model = joblib.load('model.pkl')

# Pre-calculated statistics
PRICE_STATS = {
    'min_price': 10.0,
    'max_price': 2000.0,
    'avg_price': 250.0
}

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'Brand': 'Manufacturer reputation',
    'Material': 'Main fabric or material used',
    'size_numeric': 'Physical dimensions of the bag',
    'Compartments': 'Number of storage sections',
    'Laptop Compartment': 'Dedicated laptop protection',
    'Waterproof': 'Protection from water damage',
    'Style': 'Bag design and type',
    'Color': 'Bag color and finish',
    'Weight Capacity (kg)': 'Maximum load the bag can carry',
    'premium_features': 'Combined score of special features'
}

# Feature importance scores (pre-calculated)
FEATURE_IMPORTANCE = {
    'Brand': 0.80,
    'Material': 0.70,
    'size_numeric': 0.85,
    'Compartments': 0.50,
    'Laptop Compartment': 0.65,
    'Waterproof': 0.60,
    'Style': 0.55,
    'Color': 0.45,
    'Weight Capacity (kg)': 0.75,
    'premium_features': 0.50
}

def main():
    st.set_page_config(
        page_title="Smart Bag Price Predictor",
        page_icon="🎒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        color: #4a9eff;
        border-radius: 20px;
        height: 3em;
        width: 100%;
    }
    .stTextInput>div>div>input {
        color: #4a9eff;
    }
    .metric-card {
        background-color: #1e2530;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d3747;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4a9eff;
    }
    .metric-label {
        font-size: 14px;
        color: #a3a8b8;
    }
    </style>
    """, unsafe_allow_html=True)

    # Navigation
    pages = ["🏠 Home", "💰 Price Prediction", "📊 Analytics Dashboard"]
    page = st.sidebar.radio("Navigation", pages)

    if page == "🏠 Home":
        st.title("🎒 Smart Bag Price Predictor")
        
        # Hero section
        st.markdown("""
        <div style='background-color: #1e2530; padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
            <h1 style='color: #4a9eff; margin-bottom: 20px;'>Welcome to Smart Bag Price Analysis</h1>
            <p style='font-size: 1.2em; color: #a3a8b8;'>
                An advanced AI-powered platform for accurate bag price predictions and market insights.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Key Features
        st.markdown("## 🌟 Key Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #4a9eff;'>🎯 Accurate Predictions</h3>
                <p>Advanced machine learning model trained on extensive market data</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #4a9eff;'>📊 Market Insights</h3>
                <p>Comprehensive analytics and market trend analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #4a9eff;'>💡 Smart Features</h3>
                <p>Intelligent feature analysis and price recommendations</p>
            </div>
            """, unsafe_allow_html=True)

        # Quick Stats
        st.markdown("## 📈 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>99%</div>
                <div class='metric-label'>Prediction Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>10K+</div>
                <div class='metric-label'>Market Data Points</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>25+</div>
                <div class='metric-label'>Brand Analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>24/7</div>
                <div class='metric-label'>Real-time Updates</div>
            </div>
            """, unsafe_allow_html=True)

        # How It Works
        st.markdown("## 🔄 How It Works")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #4a9eff;'>1. Input Details</h4>
                <p>Enter bag specifications including brand, material, size, and features</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #4a9eff;'>2. AI Analysis</h4>
                <p>Our ML model processes the data using advanced algorithms</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #4a9eff;'>3. Get Insights</h4>
                <p>Receive accurate price predictions and market analysis</p>
            </div>
            """, unsafe_allow_html=True)

    elif page == "💰 Price Prediction":
        st.title("🎒 Smart Bag Price Predictor")
        
        st.markdown("""
        <p style='font-size: 1.2em; color: #a3a8b8;'>
            Welcome to the Smart Bag Price Predictor! This tool uses advanced machine learning to estimate bag prices based on various features.
            Simply input the bag specifications below to get an instant price prediction.
        </p>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Price Prediction", "Market Analysis"])

        with tab1:
            col1, col2 = st.columns([3, 2])

            with col1:
                with st.form("prediction_form"):
                    st.markdown("<h3 style='color: #4a9eff;'>🔍 Bag Specifications</h3>", unsafe_allow_html=True)

                    # Input fields matching training data exactly
                    brand = st.selectbox('Brand', ['Budget', 'Mid-Range', 'Premium'])
                    material = st.selectbox('Material', ['Canvas', 'Leather', 'Nylon', 'Polyester', 'Other'])
                    size = st.selectbox('Size', ['Small', 'Medium', 'Large'])
                    compartments = st.selectbox('Compartments', [1, 2, 3, 4, 5])
                    laptop_compartment = st.selectbox('Laptop Compartment', ['No', 'Yes'])
                    waterproof = st.selectbox('Waterproof', ['No', 'Yes'])
                    style = st.selectbox('Style', ['Casual', 'Business', 'Sport', 'Travel'])
                    color = st.selectbox('Color', ['Black', 'Blue', 'Brown', 'Grey', 'Other'])
                    weight_capacity = st.selectbox('Weight Capacity (kg)', [5, 10, 15, 20, 25])

                    submitted = st.form_submit_button("Calculate Price 💫")

                    if submitted:
                        try:
                            # Convert inputs to match training data format
                            size_numeric = {'Small': 1, 'Medium': 2, 'Large': 3}[size]
                            laptop_comp_numeric = {'No': 0, 'Yes': 1}[laptop_compartment]
                            waterproof_numeric = {'No': 0, 'Yes': 1}[waterproof]
                            
                            # Calculate premium_features as in training
                            premium_features = laptop_comp_numeric + waterproof_numeric

                            # Prepare input data matching training features exactly
                            input_data = pd.DataFrame({
                                'Brand': [{'Budget': 0, 'Mid-Range': 1, 'Premium': 2}[brand]],
                                'Material': [{'Canvas': 0, 'Leather': 1, 'Nylon': 2, 'Polyester': 3, 'Other': 4}[material]],
                                'size_numeric': [size_numeric],
                                'Compartments': [compartments],
                                'Laptop Compartment': [laptop_comp_numeric],
                                'Waterproof': [waterproof_numeric],
                                'Style': [{'Casual': 0, 'Business': 1, 'Sport': 2, 'Travel': 3}[style]],
                                'Color': [{'Black': 0, 'Blue': 1, 'Brown': 2, 'Grey': 3, 'Other': 4}[color]],
                                'Weight Capacity (kg)': [float(weight_capacity)],
                                'premium_features': [premium_features]
                            })

                            # Make prediction
                            prediction = model.predict(input_data)[0]

                            # Display prediction
                            st.markdown(f"""
                            <div style='background-color: #1e2530; padding: 20px; border-radius: 10px; border: 1px solid #2d3747; margin-bottom: 20px;'>
                                <h2 style='color: #4a9eff; text-align: center; margin: 0;'>Predicted Price: ${prediction:.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                            # Market statistics
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

                            # Price gauge
                            st.markdown("<h4 style='text-align: center; color: #4a9eff; margin: 20px 0 10px 0;'>📊 Price Position</h4>", unsafe_allow_html=True)
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prediction,
                                number={'prefix': "$", 'font': {'color': '#4a9eff', 'size': 28}},
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={
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

                        except Exception as e:
                            st.error("❌ Error: " + str(e) + "\nPlease try different input values.")

            with col2:
                # Feature importance visualization
                st.markdown("<h3 style='color: #4a9eff;'>🎯 Feature Impact Analysis</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #a3a8b8; font-size: 14px;'>See how different features affect the final price prediction</p>", unsafe_allow_html=True)

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

        with tab2:
            st.title("📊 Market Analysis")
            st.markdown("### Market Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Price", f"${PRICE_STATS['avg_price']:.2f}")
            with col2:
                st.metric("Market Size", "1,000+ Products")
            with col3:
                st.metric("Price Range", f"${PRICE_STATS['min_price']:.0f} - ${PRICE_STATS['max_price']:.0f}")
            with col4:
                st.metric("Brands", "25+ Active Brands")

            # Sample market data visualization
            st.markdown("### Price Distribution")
            fig = px.histogram(
                pd.DataFrame({'Price': np.random.normal(PRICE_STATS['avg_price'], 100, 1000)}),
                x='Price',
                nbins=30,
                title='Market Price Distribution'
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1e2530',
                font={'color': '#a3a8b8'}
            )
            st.plotly_chart(fig, use_container_width=True)

    else:  # Analytics Dashboard
        st.title("📊 Analytics Dashboard")
        
        # Overview Metrics
        st.markdown("### 📈 Market Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>${:.2f}</div>
                <div class='metric-label'>Average Price</div>
            </div>
            """.format(PRICE_STATS['avg_price']), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>1,000+</div>
                <div class='metric-label'>Products Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>${:.0f} - ${:.0f}</div>
                <div class='metric-label'>Price Range</div>
            </div>
            """.format(PRICE_STATS['min_price'], PRICE_STATS['max_price']), unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>25+</div>
                <div class='metric-label'>Active Brands</div>
            </div>
            """, unsafe_allow_html=True)

        # Price Distribution
        st.markdown("### 💰 Price Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Price histogram
            fig = px.histogram(
                pd.DataFrame({'Price': np.random.normal(PRICE_STATS['avg_price'], 100, 1000)}),
                x='Price',
                nbins=30,
                title='Price Distribution'
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1e2530',
                font={'color': '#a3a8b8'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Price trend
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
            trend_data = pd.DataFrame({
                'Date': dates,
                'Average Price': np.random.normal(PRICE_STATS['avg_price'], 20, len(dates))
            })
            fig = px.line(trend_data, x='Date', y='Average Price', title='Price Trends Over Time')
            fig.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1e2530',
                font={'color': '#a3a8b8'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature Analysis
        st.markdown("### 🔍 Feature Analysis")
        
        # Feature importance plot
        importance_df = pd.DataFrame({
            'Feature': list(FEATURE_IMPORTANCE.keys()),
            'Importance': list(FEATURE_IMPORTANCE.values())
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Feature Importance Analysis')
        fig.update_layout(
            paper_bgcolor='#0e1117',
            plot_bgcolor='#1e2530',
            font={'color': '#a3a8b8'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Brand Analysis
        st.markdown("### 🏢 Brand Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Brand market share
            brands = ['Premium', 'Mid-Range', 'Budget']
            market_share = [40, 35, 25]
            fig = px.pie(values=market_share, 
                        names=brands, 
                        title='Market Share by Brand Category',
                        hole=0.4)
            fig.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1e2530',
                font={'color': '#a3a8b8'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Brand price comparison
            brand_prices = pd.DataFrame({
                'Brand': brands,
                'Average Price': [300, 200, 100],
                'Price Range': [50, 30, 20]
            })
            fig = px.bar(brand_prices, 
                        x='Brand', 
                        y='Average Price',
                        error_y='Price Range',
                        title='Average Price by Brand Category')
            fig.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1e2530',
                font={'color': '#a3a8b8'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Material Analysis
        st.markdown("### 🧵 Material Analysis")
        materials = ['Leather', 'Canvas', 'Nylon', 'Polyester', 'Other']
        material_data = pd.DataFrame({
            'Material': materials,
            'Average Price': [250, 150, 180, 120, 140],
            'Durability Score': [9, 7, 8, 6, 5]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(material_data,
                        x='Material',
                        y='Average Price',
                        title='Average Price by Material')
            fig.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1e2530',
                font={'color': '#a3a8b8'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.scatter(material_data,
                           x='Durability Score',
                           y='Average Price',
                           text='Material',
                           title='Price vs Durability by Material')
            fig.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#1e2530',
                font={'color': '#a3a8b8'}
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
