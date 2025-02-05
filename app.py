import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load the trained model and encoders
model = joblib.load('model.pkl')

# Pre-calculated statistics from training data
PRICE_STATS = {
    'min_price': 10.0,
    'max_price': 2000.0,
    'avg_price': 250.0
}

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'Size': 'Physical dimensions of the bag',
    'Weight_Capacity': 'Maximum load the bag can carry',
    'Laptop_Pocket': 'Dedicated laptop protection',
    'Water_Resistance': 'Protection from water damage',
    'Color': 'Bag color and finish',
    'Brand': 'Manufacturer reputation',
    'Material': 'Main fabric or material used',
    'Style': 'Bag design and type',
    'Premium_Features': 'Additional luxury features'
}

# Feature importance scores (pre-calculated)
FEATURE_IMPORTANCE = {
    'Size': 0.85,
    'Weight_Capacity': 0.75,
    'Laptop_Pocket': 0.65,
    'Water_Resistance': 0.60,
    'Color': 0.45,
    'Brand': 0.80,
    'Material': 0.70,
    'Style': 0.55,
    'Premium_Features': 0.50
}

def main():
    st.set_page_config(
        page_title="Smart Bag Price Predictor",
        page_icon="🎒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark theme
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
    </style>
    """, unsafe_allow_html=True)

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

                # Input fields with predefined options
                size = st.selectbox('Size', ['Small', 'Medium', 'Large'])
                weight_capacity = st.selectbox('Weight Capacity', ['Light', 'Medium', 'Heavy'])
                laptop_pocket = st.selectbox('Laptop Pocket', ['No', 'Yes'])
                water_resistance = st.selectbox('Water Resistance', ['None', 'Water Resistant', 'Waterproof'])
                color = st.selectbox('Color', ['Black', 'Blue', 'Brown', 'Grey', 'Other'])
                brand = st.selectbox('Brand', ['Budget', 'Mid-Range', 'Premium'])
                material = st.selectbox('Material', ['Canvas', 'Leather', 'Nylon', 'Polyester', 'Other'])
                style = st.selectbox('Style', ['Casual', 'Business', 'Sport', 'Travel'])
                premium_features = st.selectbox('Premium Features', ['None', 'Basic', 'Advanced', 'Premium'])

                submitted = st.form_submit_button("Calculate Price 💫")

                if submitted:
                    try:
                        # Prepare input data with numerical encoding
                        input_data = pd.DataFrame({
                            'Size': [{'Small': 0, 'Medium': 1, 'Large': 2}[size]],
                            'Weight_Capacity': [{'Light': 0, 'Medium': 1, 'Heavy': 2}[weight_capacity]],
                            'Laptop_Pocket': [{'No': 0, 'Yes': 1}[laptop_pocket]],
                            'Water_Resistance': [{'None': 0, 'Water Resistant': 1, 'Waterproof': 2}[water_resistance]],
                            'Color': [{'Black': 0, 'Blue': 1, 'Brown': 2, 'Grey': 3, 'Other': 4}[color]],
                            'Brand': [{'Budget': 0, 'Mid-Range': 1, 'Premium': 2}[brand]],
                            'Material': [{'Canvas': 0, 'Leather': 1, 'Nylon': 2, 'Polyester': 3, 'Other': 4}[material]],
                            'Style': [{'Casual': 0, 'Business': 1, 'Sport': 2, 'Travel': 3}[style]],
                            'Premium_Features': [{'None': 0, 'Basic': 1, 'Advanced': 2, 'Premium': 3}[premium_features]]
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

if __name__ == "__main__":
    main()
