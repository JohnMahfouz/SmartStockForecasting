import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import datetime
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SmartStock AI - Egypt", 
    page_icon="🛒", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS AND DATA
# =============================================================================

@st.cache_resource
def load_models():
    """Load both store and product models"""
    try:
        model_store = xgb.XGBRegressor()
        model_store.load_model('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\smartstock_store_model.json')
        
        model_product = xgb.XGBRegressor()
        model_product.load_model('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\smartstock_product_model.json')
        
        with open('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\model_features.json', 'r') as f:
            features = json.load(f)
        
        with open('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        return model_store, model_product, features, metrics
    except FileNotFoundError as e:
        st.error(f"""
        ❌ **Model files not found!**
        
        Missing file: `{str(e)}`
        
        **Required files:**
        - `smartstock_store_model.json`
        - `smartstock_product_model.json`
        - `model_features.json`
        - `model_metrics.json`
        
        **To fix this:**
        1. Make sure you ran `python Model_Train.py` successfully
        2. Check that the model files are in the same directory as app.py
        3. If training failed, check the error messages in Model_Train.py
        
        **Current directory:** {os.getcwd()}
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

@st.cache_data
def load_historical_data():
    """Load historical data for context"""
    try:
        df_store = pd.read_csv('egyptian_sales_store_level.csv', parse_dates=['Date'])
        df_product = pd.read_csv('egyptian_sales_product_level.csv', parse_dates=['Date'])
        return df_store, df_product
    except FileNotFoundError:
        return None, None

model_store, model_product, feature_config, metrics = load_models()
df_store_hist, df_product_hist = load_historical_data()

# =============================================================================
# PRODUCT CONFIGURATION
# =============================================================================

PRODUCTS = {
    "Bakery (Fresh Bread)": {
        "category": "Bakery",
        "share_of_revenue": 0.08,
        "price": 35,
        "shelf_life": 1,
        "risk_type": "🔴 HIGH WASTE",
        "ramadan_effect": "↓ Decreases (fasting)",
        "summer_effect": "↓ Slight decrease",
        "storage_temp": "Room temp",
        "best_days": "Thursday-Friday"
    },
    "Dairy (Yogurt/Milk)": {
        "category": "Dairy",
        "share_of_revenue": 0.15,
        "price": 60,
        "shelf_life": 12,
        "risk_type": "🟡 MEDIUM RISK",
        "ramadan_effect": "↑↑ 2.5x increase (Suhoor)",
        "summer_effect": "↓ Heat sensitivity",
        "storage_temp": "4°C refrigerated",
        "best_days": "Weekend + Payday"
    },
    "Beverages (Juice/Soda)": {
        "category": "Beverages",
        "share_of_revenue": 0.10,
        "price": 25,
        "shelf_life": 180,
        "risk_type": "🟢 LOW RISK",
        "ramadan_effect": "↑ 1.8x increase",
        "summer_effect": "↑↑ High demand",
        "storage_temp": "Room temp",
        "best_days": "Summer weekends"
    },
    "Fresh Produce (Vegetables)": {
        "category": "Fresh_Produce",
        "share_of_revenue": 0.12,
        "price": 20,
        "shelf_life": 4,
        "risk_type": "🔴 HIGH WASTE",
        "ramadan_effect": "↑ 1.5x increase",
        "summer_effect": "↓ Spoils faster",
        "storage_temp": "Cool storage",
        "best_days": "Thursday (pre-weekend)"
    },
    "Frozen Foods": {
        "category": "Frozen_Foods",
        "share_of_revenue": 0.09,
        "price": 45,
        "shelf_life": 180,
        "risk_type": "🟢 LOW RISK",
        "ramadan_effect": "↑ Moderate increase",
        "summer_effect": "→ Stable",
        "storage_temp": "-18°C freezer",
        "best_days": "Weekend shopping"
    },
    "Snacks": {
        "category": "Snacks",
        "share_of_revenue": 0.11,
        "price": 15,
        "shelf_life": 90,
        "risk_type": "🟢 LOW RISK",
        "ramadan_effect": "↑↑ 2x increase (Iftar)",
        "summer_effect": "↑ Slight increase",
        "storage_temp": "Room temp",
        "best_days": "Ramadan + weekends"
    },
    "Household Items": {
        "category": "Household",
        "share_of_revenue": 0.14,
        "price": 50,
        "shelf_life": 365,
        "risk_type": "🟢 VERY LOW",
        "ramadan_effect": "↑ Pre-Ramadan spike",
        "summer_effect": "→ Stable",
        "storage_temp": "Room temp",
        "best_days": "Payday periods"
    },
    "Personal Care": {
        "category": "Personal_Care",
        "share_of_revenue": 0.10,
        "price": 70,
        "shelf_life": 540,
        "risk_type": "🟢 VERY LOW",
        "ramadan_effect": "↓ Slight decrease",
        "summer_effect": "↑ Hygiene products",
        "storage_temp": "Room temp",
        "best_days": "Weekend + payday"
    }
}

# =============================================================================
# SIDEBAR - INPUT CONTROLS
# =============================================================================

st.sidebar.markdown("# 🛒 SmartStock AI")
st.sidebar.markdown("### Egyptian Market Forecasting")
st.sidebar.markdown("---")

# Date Selection
st.sidebar.subheader("📅 Forecast Date")
input_date = st.sidebar.date_input(
    "Select date",
    datetime.date(2026, 1, 14),
    min_value=datetime.date(2024, 1, 1),
    max_value=datetime.date(2027, 12, 31)
)

# Auto-detect day features
day_of_week = input_date.weekday()
day_name = input_date.strftime('%A')
is_weekend = 1 if day_of_week in [4, 5] else 0  # Friday, Saturday
is_thursday = 1 if day_of_week == 3 else 0

# Ramadan detection
def is_ramadan_date(date):
    """Check if date falls in Ramadan"""
    ramadan_ranges = [
        (datetime.date(2024, 3, 11), datetime.date(2024, 4, 9)),
        (datetime.date(2025, 3, 1), datetime.date(2025, 3, 30)),
        (datetime.date(2026, 2, 18), datetime.date(2026, 3, 19))
    ]
    for start, end in ramadan_ranges:
        if start <= date <= end:
            return True
    return False

is_ramadan_auto = is_ramadan_date(input_date)

# Context toggles
st.sidebar.subheader("🎯 Market Context")
is_promo = st.sidebar.toggle("💰 White Friday / Promotion", value=False)
is_ramadan = st.sidebar.toggle("🌙 Ramadan Period", value=is_ramadan_auto)
is_payday = st.sidebar.toggle(
    "💸 Payday Period", 
    value=(input_date.day <= 5 or input_date.day >= 28)
)

# Display day info
st.sidebar.info(f"""
**Date:** {day_name}, {input_date.strftime('%B %d, %Y')}  
**Weekend:** {'✅ Yes' if is_weekend else '❌ No'}  
**Ramadan:** {'✅ Yes' if is_ramadan else '❌ No'}  
**Payday:** {'✅ Yes' if is_payday else '❌ No'}
""")

# Historical sales context
st.sidebar.subheader("📊 Recent Sales Data")
st.sidebar.markdown("*Enter yesterday's and last week's store revenue*")

lag_1 = st.sidebar.number_input(
    "Yesterday's Revenue (EGP)",
    min_value=10000,
    max_value=150000,
    value=45000,
    step=1000
)

lag_7 = st.sidebar.number_input(
    "Last Week Same Day (EGP)",
    min_value=10000,
    max_value=150000,
    value=42000,
    step=1000
)

# Product selection
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Product Category")
selected_product = st.sidebar.selectbox(
    "Select product to forecast:",
    list(PRODUCTS.keys())
)

product_profile = PRODUCTS[selected_product]

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

# Header
st.markdown(f'<div class="main-header">🛒 SmartStock AI - Egypt Edition</div>', unsafe_allow_html=True)

# Product overview
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📦 Category", product_profile['category'])
with col2:
    st.metric("💵 Price", f"{product_profile['price']} EGP")
with col3:
    st.metric("⏰ Shelf Life", f"{product_profile['shelf_life']} days")
with col4:
    st.metric("⚠️ Risk", product_profile['risk_type'])

st.markdown("---")

# Market Insights
with st.expander("📊 Product Market Intelligence", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Ramadan Impact:** {product_profile['ramadan_effect']}  
        **Summer Impact:** {product_profile['summer_effect']}  
        """)
    with col2:
        st.markdown(f"""
        **Storage:** {product_profile['storage_temp']}  
        **Best Days:** {product_profile['best_days']}
        """)

st.markdown("---")

# =============================================================================
# PREDICTION BUTTON
# =============================================================================

if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
    
    with st.spinner("🔮 Analyzing market conditions and generating forecast..."):
        
        # =====================================================================
        # STEP 1: PREPARE FEATURES
        # =====================================================================
        
        # Calculate additional lag features
        lag_2 = lag_1 * 0.98  # Estimate
        lag_3 = lag_1 * 0.97
        lag_14 = lag_7 * 1.02
        lag_28 = lag_7 * 0.95
        
        # Rolling statistics (estimates)
        rolling_mean_7 = (lag_1 + lag_7 * 6) / 7
        rolling_mean_14 = lag_7
        rolling_mean_28 = lag_7 * 0.98
        
        rolling_std_7 = abs(lag_1 - lag_7) * 0.5
        rolling_std_14 = rolling_std_7 * 1.2
        rolling_std_28 = rolling_std_7 * 1.5
        
        # Growth rates
        growth_7d = (lag_1 - lag_7) / (lag_7 + 1)
        growth_28d = (lag_1 - lag_28) / (lag_28 + 1)
        
        # Day-of-week and month averages (from historical data if available)
        if df_store_hist is not None:
            dow_avg = df_store_hist[df_store_hist['DayOfWeek'] == day_of_week]['Revenue_EGP'].mean()
            month_avg = df_store_hist[df_store_hist['Month'] == input_date.month]['Revenue_EGP'].mean()
        else:
            dow_avg = lag_7
            month_avg = lag_7
        
        # Ramadan phase
        if is_ramadan:
            # Simplified: assume mid-Ramadan
            ramadan_phase = 2
        else:
            ramadan_phase = 0
        
        # Eid flags (simplified)
        is_eid_fitr = 0
        is_eid_adha = 0
        
        # Seasonal flags
        is_summer = 1 if input_date.month in [6, 7, 8] else 0
        is_winter = 1 if input_date.month in [12, 1, 2] else 0
        is_school = 1 if input_date.month not in [6, 7, 8] else 0
        
        # =====================================================================
        # STEP 2: STORE-LEVEL PREDICTION
        # =====================================================================
        
        store_input = pd.DataFrame({
            'Is_White_Friday': [1 if is_promo else 0],
            'Is_Weekend': [is_weekend],
            'Is_Thursday': [is_thursday],
            'Ramadan_Phase': [ramadan_phase],
            'Is_Ramadan': [1 if is_ramadan else 0],
            'Is_Eid_Fitr': [is_eid_fitr],
            'Is_Eid_Adha': [is_eid_adha],
            'Is_Coptic_Christmas': [1 if (input_date.month == 1 and input_date.day == 7) else 0],
            'Is_Payday': [1 if is_payday else 0],
            'Is_Payday_Early': [1 if input_date.day <= 5 else 0],
            'Is_Payday_Late': [1 if input_date.day >= 28 else 0],
            'Is_Summer': [is_summer],
            'Is_Winter': [is_winter],
            'Is_School_Season': [is_school],
            'DayOfWeek': [day_of_week],
            'Month': [input_date.month],
            'Day': [input_date.day],
            'Lag_1': [lag_1],
            'Lag_2': [lag_2],
            'Lag_3': [lag_3],
            'Lag_7': [lag_7],
            'Lag_14': [lag_14],
            'Lag_28': [lag_28],
            'Rolling_Mean_7': [rolling_mean_7],
            'Rolling_Mean_14': [rolling_mean_14],
            'Rolling_Mean_28': [rolling_mean_28],
            'Rolling_Std_7': [rolling_std_7],
            'Rolling_Std_14': [rolling_std_14],
            'Rolling_Std_28': [rolling_std_28],
            'DOW_Avg': [dow_avg],
            'Month_Avg': [month_avg],
            'Growth_Rate_7d': [growth_7d],
            'Growth_Rate_28d': [growth_28d]
        })
        
        # Predict store revenue
        store_revenue_pred = model_store.predict(store_input)[0]
        store_revenue_pred = max(store_revenue_pred, 0)  # No negative predictions
        
        # =====================================================================
        # STEP 3: PRODUCT-LEVEL ALLOCATION
        # =====================================================================
        
        # Base product revenue
        product_revenue = store_revenue_pred * product_profile['share_of_revenue']
        
        # Apply contextual adjustments (from product profile)
        adjustments = []
        
        if is_ramadan:
            if product_profile['category'] == 'Dairy':
                product_revenue *= 2.5
                adjustments.append("🌙 Ramadan boost: +150%")
            elif product_profile['category'] == 'Snacks':
                product_revenue *= 2.0
                adjustments.append("🌙 Ramadan boost: +100%")
            elif product_profile['category'] == 'Beverages':
                product_revenue *= 1.8
                adjustments.append("🌙 Ramadan boost: +80%")
            elif product_profile['category'] == 'Fresh_Produce':
                product_revenue *= 1.5
                adjustments.append("🌙 Ramadan boost: +50%")
            elif product_profile['category'] == 'Bakery':
                product_revenue *= 0.8
                adjustments.append("🌙 Ramadan reduction: -20%")
        
        if is_summer:
            if product_profile['category'] == 'Beverages':
                product_revenue *= 1.4
                adjustments.append("☀️ Summer boost: +40%")
            elif product_profile['category'] in ['Bakery', 'Fresh_Produce', 'Dairy']:
                product_revenue *= 0.9
                adjustments.append("☀️ Summer heat: -10%")
        
        if is_weekend:
            if product_profile['category'] in ['Fresh_Produce', 'Snacks', 'Beverages']:
                product_revenue *= 1.2
                adjustments.append("📅 Weekend boost: +20%")
        
        if is_promo:
            product_revenue *= 1.5
            adjustments.append("💰 Promotion boost: +50%")
        
        # Calculate units
        units_forecast = int(product_revenue / product_profile['price'])
        units_forecast = max(units_forecast, 0)
        
        # =====================================================================
        # STEP 4: RISK-ADJUSTED RECOMMENDATION
        # =====================================================================
        
        shelf_life = product_profile['shelf_life']
        
        if shelf_life <= 2:
            # High waste risk - order conservatively
            safety_factor = 0.85
            rec_order = int(units_forecast * safety_factor)
            risk_level = "HIGH"
            risk_color = "alert-high"
            risk_message = f"⚠️ **HIGH WASTE RISK**: Shelf life is only {shelf_life} day(s). Order conservatively at 85% of forecast."
        elif shelf_life <= 7:
            # Medium risk
            safety_factor = 0.92
            rec_order = int(units_forecast * safety_factor)
            risk_level = "MEDIUM"
            risk_color = "alert-medium"
            risk_message = f"⚡ **MEDIUM RISK**: {shelf_life} days shelf life. Slight safety buffer applied (92% of forecast)."
        elif shelf_life <= 30:
            # Low-medium risk
            safety_factor = 1.0
            rec_order = units_forecast
            risk_level = "LOW-MEDIUM"
            risk_color = "alert-low"
            risk_message = f"✅ **LOW-MEDIUM RISK**: {shelf_life} days shelf life. Order full forecast amount."
        else:
            # Very low risk - can over-order
            safety_factor = 1.10
            rec_order = int(units_forecast * safety_factor)
            risk_level = "VERY LOW"
            risk_color = "alert-low"
            risk_message = f"✅ **VERY LOW RISK**: {shelf_life}+ days shelf life. Safe to order 110% for buffer stock."
        
        # =====================================================================
        # STEP 5: DISPLAY RESULTS
        # =====================================================================
        
        st.success("✅ Forecast generated successfully!")
        
        # Key metrics
        st.markdown("### 📊 Forecast Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🏪 Store Revenue Forecast",
                f"{store_revenue_pred:,.0f} EGP",
                delta=f"{((store_revenue_pred - lag_1) / lag_1 * 100):+.1f}% vs yesterday"
            )
        
        with col2:
            st.metric(
                "📦 Product Revenue",
                f"{product_revenue:,.0f} EGP"
            )
        
        with col3:
            st.metric(
                "🎯 Raw Forecast",
                f"{units_forecast:,} units"
            )
        
        with col4:
            st.metric(
                "✅ Recommended Order",
                f"{rec_order:,} units",
                delta=f"Safety: {safety_factor:.0%}"
            )
        
        # Risk assessment
        st.markdown("### ⚠️ Risk Assessment")
        st.markdown(f'<div class="{risk_color}">{risk_message}</div>', unsafe_allow_html=True)
        
        # Applied adjustments
        if adjustments:
            st.markdown("### 🔧 Applied Adjustments")
            for adj in adjustments:
                st.markdown(f"- {adj}")
        
        # Model confidence
        st.markdown("### 📈 Forecast Confidence")
        store_mape = metrics['store_model']['test_mape']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy (MAPE)", f"{100 - store_mape:.1f}%")
            confidence = "High" if store_mape < 10 else "Medium" if store_mape < 15 else "Low"
            st.info(f"**Confidence Level:** {confidence}")
        
        with col2:
            error_margin = int(store_revenue_pred * (store_mape / 100))
            st.metric("Error Margin (±)", f"{error_margin:,} EGP")
            st.info(f"Forecast range: {store_revenue_pred - error_margin:,.0f} - {store_revenue_pred + error_margin:,.0f} EGP")
        
        # =====================================================================
        # STEP 6: VISUALIZATION
        # =====================================================================
        
        if df_store_hist is not None:
            st.markdown("### 📊 Historical Context")
            
            # Get last 30 days
            recent = df_store_hist.tail(30).copy()
            recent['Prediction'] = None
            
            # Add prediction point
            pred_row = pd.DataFrame({
                'Date': [pd.Timestamp(input_date)],
                'Revenue_EGP': [None],
                'Prediction': [store_revenue_pred]
            })
            
            chart_data = pd.concat([recent[['Date', 'Revenue_EGP']], pred_row])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=recent['Date'],
                y=recent['Revenue_EGP'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(input_date)],
                y=[store_revenue_pred],
                mode='markers',
                name='Forecast',
                marker=dict(color='#ff7f0e', size=15, symbol='star')
            ))
            
            fig.update_layout(
                title="Store Revenue - Last 30 Days + Forecast",
                xaxis_title="Date",
                yaxis_title="Revenue (EGP)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # STEP 7: RECOMMENDATIONS
        # =====================================================================
        
        st.markdown("### 💡 Action Items")
        
        recommendations = []
        
        # Ramadan recommendations
        if is_ramadan:
            if product_profile['category'] in ['Dairy', 'Snacks', 'Beverages']:
                recommendations.append("🌙 **Ramadan Peak**: This is a high-demand product. Ensure adequate stock before Iftar hours.")
            else:
                recommendations.append("🌙 **Ramadan Period**: Monitor demand closely as patterns differ from normal days.")
        
        # Shelf life recommendations
        if shelf_life <= 2:
            recommendations.append(f"⚠️ **Perishable Item**: Only {shelf_life} day shelf life. Order for next-day delivery only.")
            recommendations.append("📍 **Quality Check**: Inspect goods on arrival and monitor temperature strictly.")
        
        # Weekend recommendations
        if is_weekend:
            recommendations.append("📅 **Weekend**: Higher foot traffic expected. Ensure adequate staffing.")
        
        # Payday recommendations
        if is_payday:
            recommendations.append("💸 **Payday Period**: Increased spending power. Consider premium product placement.")
        
        # Summer recommendations
        if is_summer and product_profile['category'] in ['Fresh_Produce', 'Dairy', 'Bakery']:
            recommendations.append("☀️ **Summer Heat**: Spoilage risk elevated. Check cold chain and storage conditions.")
        
        # General recommendation
        recommendations.append(f"📦 **Order Quantity**: Place order for **{rec_order:,} units** ({rec_order * product_profile['price']:,.0f} EGP)")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")

# =============================================================================
# FOOTER - MODEL INFO
# =============================================================================

st.markdown("---")
with st.expander("ℹ️ About This System"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **SmartStock AI** uses machine learning to forecast retail demand for Egyptian markets.
        
        **Features:**
        - Store-level revenue prediction
        - Product-specific forecasting
        - Egyptian calendar integration (Ramadan, Eids, etc.)
        - Risk-adjusted ordering recommendations
        - Shelf life considerations
        
        **Model:** XGBoost Gradient Boosting
        """)
    
    with col2:
        st.markdown(f"""
        **Model Performance:**
        - Store Model MAPE: {metrics['store_model']['test_mape']:.2f}%
        - Store Model R²: {metrics['store_model']['test_r2']:.4f}
        - Product Model MAPE: {metrics['product_model']['test_mape']:.2f}%
        
        **Data Coverage:**
        - Training period: 2024-2025
        - Egyptian market patterns
        - {len(PRODUCTS)} product categories
        """)

st.markdown("---")
