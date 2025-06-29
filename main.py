import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go


# === Insights Generation Functions ===
def generate_wqi_insights(wqi_value, location, period_description):
    """Generate insights based on WQI value"""
    insights = []

    if wqi_value <= 25:
        insights.extend([
            f"🚨 **Critical Alert**: {location} shows poor water quality (WQI: {wqi_value:.1f}) during {period_description.lower()}.",
            "🐟 **Marine Life Impact**: Fish populations are likely stressed, with potential fish kills and disrupted breeding cycles.",
            "⚡ **Immediate Actions**: Implement emergency water treatment, restrict fishing activities, and investigate pollution sources.",
            "🌊 **Taal Lake Specific**: The volcanic nature of Taal Lake makes it particularly sensitive to chemical imbalances."
        ])
    elif wqi_value <= 50:
        insights.extend([
            f"⚠️ **Caution**: {location} has fair water quality (WQI: {wqi_value:.1f}) requiring attention.",
            "🐟 **Marine Life Impact**: Some fish species may experience reduced growth rates and reproductive success.",
            "🔧 **Recommended Actions**: Increase monitoring frequency, implement water treatment measures, and control nutrient inputs.",
            "🌋 **Taal Consideration**: Monitor volcanic activity as it can compound water quality issues."
        ])
    elif wqi_value <= 70:
        insights.extend([
            f"📊 **Moderate Status**: {location} maintains average water quality (WQI: {wqi_value:.1f}).",
            "🐟 **Marine Life Impact**: Most fish species can survive but optimal conditions are not met for thriving ecosystems.",
            "📈 **Improvement Actions**: Focus on reducing nutrient pollution, enhance waste management around the lake.",
            "🏞️ **Tourism Impact**: Water quality may affect recreational activities and eco-tourism potential."
        ])
    elif wqi_value <= 90:
        insights.extend([
            f"✅ **Good Status**: {location} demonstrates good water quality (WQI: {wqi_value:.1f}).",
            "🐟 **Marine Life Impact**: Fish populations can thrive with healthy breeding and growth conditions.",
            "🔄 **Maintenance Actions**: Continue current management practices and regular monitoring to prevent degradation.",
            "🌟 **Taal Success**: This reflects positive lake management and controlled human activities around Taal."
        ])
    else:
        insights.extend([
            f"🌟 **Excellent Status**: {location} achieves outstanding water quality (WQI: {wqi_value:.1f})!",
            "🐟 **Marine Life Impact**: Optimal conditions for diverse aquatic ecosystems and fish populations.",
            "🏆 **Best Practices**: Serve as a model for other areas of Taal Lake and maintain current conservation efforts.",
            "🌍 **Environmental Success**: Demonstrates excellent balance between human activities and ecosystem protection."
        ])

    return insights


def generate_pollutant_insights(analysis_data, classifications, location):
    """Generate insights based on pollutant levels"""
    insights = []

    # Ammonia insights
    ammonia_val = analysis_data['ammonia']
    if classifications['ammonia_class'] == 'Poor':
        insights.extend([
            f"🧪 **Ammonia Crisis**: High ammonia levels ({ammonia_val:.3f} mg/L) in {location} are toxic to fish.",
            "🐟 **Fish Impact**: Ammonia poisoning can cause gill damage, reduced oxygen uptake, and fish mortality.",
            "💧 **Source Control**: Investigate sewage discharge, agricultural runoff, and decomposing organic matter.",
            "🌊 **Taal Action**: Implement immediate water circulation and aeration systems if possible."
        ])
    elif classifications['ammonia_class'] == 'Moderate':
        insights.extend([
            f"⚠️ **Ammonia Watch**: Moderate ammonia levels ({ammonia_val:.3f} mg/L) require monitoring.",
            "🐟 **Fish Stress**: Prolonged exposure may cause stress and reduced immunity in fish populations.",
            "🔍 **Prevention**: Monitor waste discharge points and implement better waste treatment systems."
        ])
    else:
        insights.append(
            f"✅ **Ammonia Safe**: Good ammonia levels ({ammonia_val:.3f} mg/L) support healthy aquatic life.")

    # Nitrate insights
    nitrate_val = analysis_data['nitrate']
    if classifications['nitrate_class'] == 'High':
        insights.extend([
            f"🌱 **Nitrate Overload**: High nitrate levels ({nitrate_val:.2f} mg/L) promote excessive algae growth.",
            "🌊 **Eutrophication Risk**: May lead to algal blooms, oxygen depletion, and fish kills in Taal Lake.",
            "🚜 **Agricultural Impact**: Likely from fertilizer runoff from surrounding agricultural areas.",
            "🏘️ **Community Action**: Engage local farmers in sustainable farming practices and buffer zones."
        ])
    elif classifications['nitrate_class'] == 'Elevated':
        insights.extend([
            f"📈 **Nitrate Rising**: Elevated nitrate levels ({nitrate_val:.2f} mg/L) show early eutrophication signs.",
            "🌿 **Algae Growth**: Increased plant growth may start affecting water clarity and oxygen levels.",
            "🔄 **Management**: Implement nutrient management strategies before conditions worsen."
        ])
    else:
        insights.append(
            f"✅ **Nitrate Controlled**: Safe nitrate levels ({nitrate_val:.2f} mg/L) prevent excessive plant growth.")

    # Phosphate insights
    pho_val = analysis_data['pho']
    if classifications['pho_class'] == 'High':
        insights.extend([
            f"🧪 **Phosphate Excess**: High phosphate levels ({pho_val:.2f} mg/L) accelerate eutrophication.",
            "🌊 **Lake Ecosystem**: Combined with nitrates, creates perfect conditions for harmful algal blooms.",
            "🏠 **Detergent Impact**: Often from household detergents and industrial discharge around Taal.",
            "📋 **Regulation**: Enforce stricter wastewater treatment and phosphate-free detergent policies."
        ])
    elif classifications['pho_class'] == 'Low':
        insights.extend([
            f"📉 **Phosphate Deficiency**: Low phosphate levels ({pho_val:.2f} mg/L) may limit plant growth.",
            "⚖️ **Balance Needed**: While preventing algal blooms, some phosphate is needed for healthy ecosystems.",
            "🔄 **Natural Cycle**: Monitor if this is natural variation or due to management interventions."
        ])
    else:
        insights.append(
            f"✅ **Phosphate Balanced**: Optimal phosphate levels ({pho_val:.2f} mg/L) support healthy plant growth.")

    return insights


def generate_taal_specific_recommendations(wqi_value, analysis_data, location):
    """Generate Taal Lake-specific recommendations"""
    recommendations = []

    # General Taal Lake context
    recommendations.extend([
        "🌋 **Volcanic Lake Management**: Taal Lake's unique volcanic environment requires specialized monitoring approaches.",
        "🏞️ **Tourism Balance**: Maintain water quality to support both ecological health and sustainable tourism industry."
    ])

    # Specific recommendations based on conditions
    if wqi_value < 70:
        recommendations.extend([
            "🚤 **Boat Regulations**: Implement stricter controls on motorboat operations to reduce fuel contamination.",
            "🏘️ **Community Sewage**: Upgrade sewage treatment facilities in lakeside communities (Talisay, Balete, etc.).",
            "🐄 **Livestock Management**: Control cattle waste from farms that may drain into the lake.",
            "🌾 **Agricultural Buffer**: Create buffer zones between agricultural areas and the lake shoreline."
        ])

    if analysis_data['ammonia'] > 0.1 or analysis_data['nitrate'] > 15:
        recommendations.extend([
            "🔬 **Hotspot Identification**: Map specific pollution sources around the lake perimeter.",
            "💧 **Water Circulation**: Consider artificial aeration systems in heavily polluted areas.",
            "🤝 **Community Engagement**: Work with local barangays on waste management education."
        ])

    # Positive reinforcement for good conditions
    if wqi_value > 70:
        recommendations.extend([
            "📊 **Monitoring Network**: Expand current monitoring to maintain these good conditions.",
            "🌟 **Best Practice Sharing**: Share successful management strategies with other Philippine lakes.",
            "🎣 **Sustainable Fishing**: Promote sustainable fishing practices to maintain healthy fish populations."
        ])

    recommendations.extend([
        "📱 **Real-time Monitoring**: Implement IoT sensors for continuous water quality monitoring.",
        "🏛️ **Policy Integration**: Coordinate with DENR and local government units for comprehensive lake management.",
        "🌊 **Climate Adaptation**: Prepare for climate change impacts on lake water quality and volcanic activity."
    ])

    return recommendations


# === Main Insights Display Function ===
def display_insights_section(wqi_value, analysis_data, classifications, location, period_description):
    """Display comprehensive insights section"""

    st.markdown("---")  # Separator line
    st.header("🔍 Water Quality Insights & Recommendations")
    st.markdown(f"*Analysis for {location} based on {period_description.lower()}*")

    # Create tabs for different types of insights
    tab1, tab2, tab3 = st.tabs(["📊 Overall Assessment", "🧪 Pollutant Analysis", "🌊 Taal Lake Management"])

    with tab1:
        st.subheader("Water Quality Index Assessment")
        wqi_insights = generate_wqi_insights(wqi_value, location, period_description)

        for insight in wqi_insights:
            if insight.startswith("🚨") or insight.startswith("⚠️"):
                st.error(insight)
            elif insight.startswith("✅") or insight.startswith("🌟"):
                st.success(insight)
            else:
                st.info(insight)

        # Add summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Water Quality Grade",
                      "A" if wqi_value > 90 else "B" if wqi_value > 70 else "C" if wqi_value > 50 else "D" if wqi_value > 25 else "F",
                      help=f"Based on WQI of {wqi_value:.1f}")
        with col2:
            ecosystem_health = "Thriving" if wqi_value > 80 else "Stable" if wqi_value > 60 else "Stressed" if wqi_value > 40 else "Critical"
            st.metric("Ecosystem Health", ecosystem_health)
        with col3:
            tourism_impact = "Excellent" if wqi_value > 80 else "Good" if wqi_value > 60 else "Moderate" if wqi_value > 40 else "Poor"
            st.metric("Tourism Suitability", tourism_impact)

    with tab2:
        st.subheader("Pollutant-Specific Analysis")
        pollutant_insights = generate_pollutant_insights(analysis_data, classifications, location)

        for insight in pollutant_insights:
            if "Crisis" in insight or "Overload" in insight:
                st.error(insight)
            elif "Watch" in insight or "Rising" in insight or "Deficiency" in insight:
                st.warning(insight)
            elif "Safe" in insight or "Controlled" in insight or "Balanced" in insight:
                st.success(insight)
            else:
                st.info(insight)

        # Pollution risk assessment
        st.subheader("🚨 Pollution Risk Assessment")
        risk_factors = []
        if analysis_data['ammonia'] > 0.2: risk_factors.append("High Ammonia")
        if analysis_data['nitrate'] > 15: risk_factors.append("Elevated Nitrate")
        if analysis_data['pho'] > 4.0: risk_factors.append("High Phosphate")
        if analysis_data['ph'] < 6.5 or analysis_data['ph'] > 8.5: risk_factors.append("pH Imbalance")

        if risk_factors:
            st.error(f"⚠️ **Active Risk Factors**: {', '.join(risk_factors)}")
        else:
            st.success("✅ **No Major Risk Factors Detected**")

    with tab3:
        st.subheader("Taal Lake-Specific Management Recommendations")
        taal_recommendations = generate_taal_specific_recommendations(wqi_value, analysis_data, location)

        for i, recommendation in enumerate(taal_recommendations, 1):
            st.markdown(f"{recommendation}")

        # Priority actions
        st.subheader("🎯 Priority Actions")
        if wqi_value < 50:
            priority_level = "🚨 **URGENT**"
            priority_color = "red"
            priority_actions = [
                "Immediate source identification and pollution control",
                "Emergency response plan activation",
                "Public health advisory consideration"
            ]
        elif wqi_value < 70:
            priority_level = "⚠️ **HIGH**"
            priority_color = "orange"
            priority_actions = [
                "Enhanced monitoring and treatment systems",
                "Community engagement and education",
                "Agricultural runoff control measures"
            ]
        else:
            priority_level = "✅ **MAINTENANCE**"
            priority_color = "green"
            priority_actions = [
                "Continue current best practices",
                "Preventive monitoring and maintenance",
                "Sustainable development planning"
            ]

        st.markdown(f"**Priority Level**: {priority_level}")
        for action in priority_actions:
            st.markdown(f"• {action}")

# === Load CSV Data ===
@st.cache_data
def load_csv_data():
    """Load and preprocess the CSV data"""
    try:
        # CSV file in the same directory as main.py
        csv_path = "./water_quality_data.csv"
        df = pd.read_csv(csv_path)

        # Convert date column to datetime (adjust format as needed)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'year' in df.columns and 'month' in df.columns:
            # If you have separate year and month columns
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

        # Sort by location and date
        df = df.sort_values(['location', 'date'])

        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None


# === Load assets (keep existing model loading for potential future use) ===
INPUT_SCALER_PATH = "./input_scaler_std.save"
TARGET_SCALER_PATH = "./target_scaler_std.save"
MODEL_PATH = "./hybrid_model_v2.keras"
INPUT_FEATURES = ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION', 'co2', 'so2']
TARGET_FEATURES = ['temperature', 'ph', 'ammonia', 'nitrate', 'pho', 'dissolved_oxygen']

# Try to load models (optional, for future predictions)
try:
    model = load_model(MODEL_PATH)
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    models_loaded = True
except:
    st.warning("Model files not found. Using CSV data only.")
    models_loaded = False


# === Enhanced WQI Gauge Functions (keep existing) ===
def create_wqi_gauge(wqi_value):
    """
    Create a speedometer-style gauge for Water Quality Index (WQI) with the current range elevated
    and a clean legend on the side
    """
    # Define WQI ranges and colors
    wqi_ranges = {
        'Poor': {'range': [0, 25], 'color': '#FF4444'},
        'Fair': {'range': [26, 50], 'color': '#FF8C00'},
        'Average': {'range': [51, 70], 'color': '#FFD700'},
        'Good': {'range': [71, 90], 'color': '#90EE90'},
        'Excellent': {'range': [91, 100], 'color': '#008000'}
    }

    # Determine current WQI category
    def get_wqi_category(value):
        for category, info in wqi_ranges.items():
            if info['range'][0] <= value <= info['range'][1]:
                return category, info['color']
        return 'Unknown', '#808080'

    current_category, current_color = get_wqi_category(wqi_value)

    # Create gauge steps with elevation effect for current range
    steps = []
    for category, info in wqi_ranges.items():
        if info['range'][0] <= wqi_value <= info['range'][1]:
            steps.append({
                'range': info['range'],
                'color': info['color'],
                'line': {'color': "#90EE90", 'width': 4},
                'thickness': 0.8
            })
        else:
            steps.append({
                'range': info['range'],
                'color': info['color'],
                'line': {'color': "#CCCCCC", 'width': 1},
                'thickness': 0.6
            })

    # Create the gauge
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=wqi_value,
        domain={'x': [0, 0.7], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 2,
                'tickcolor': "darkgray",
                'tickfont': {'color': "darkgray", 'size': 12, 'family': "Arial"}
            },
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#666666",
            'steps': steps
        }
    ))

    # Add legend
    legend_items = [
        {"text": "Excellent", "range": "91-100", "color": "#008000"},
        {"text": "Good", "range": "71-90", "color": "#90EE90"},
        {"text": "Average", "range": "51-70", "color": "#FFD700"},
        {"text": "Fair", "range": "26-50", "color": "#FF8C00"},
        {"text": "Poor", "range": "0-25", "color": "#FF4444"}
    ]

    annotations = []
    for i, item in enumerate(legend_items):
        y_position = 0.8 - (i * 0.15)
        is_current = item["text"].lower() == current_category.lower()

        annotations.append(
            dict(
                x=0.75,
                y=y_position,
                text="■",
                showarrow=False,
                font=dict(size=20, color=item["color"]),
                xanchor="left"
            )
        )

        text_style = dict(size=14, family="Arial Black") if is_current else dict(size=12, family="Arial")
        text_color = "black" if is_current else "#666666"

        annotations.append(
            dict(
                x=0.78,
                y=y_position,
                text=f"<b>{item['text']}</b> ({item['range']})" if is_current else f"{item['text']} ({item['range']})",
                showarrow=False,
                font=dict(color=text_color, **text_style),
                xanchor="left"
            )
        )

    fig.update_layout(
        height=450,
        width=600,
        annotations=annotations,
        margin=dict(l=50, r=150, t=50, b=50),
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# === Parameter-specific gauge functions (keep existing) ===
def create_ammonia_gauge(ammonia_value):
    """Create gauge for ammonia levels"""
    ranges = {
        'Good': {'range': [0, 0.05], 'color': '#008000'},
        'Moderate': {'range': [0.05, 0.5], 'color': '#FFD700'},
        'Poor': {'range': [0.5, 1.0], 'color': '#FF4444'}
    }

    current_category = 'Good' if ammonia_value <= 0.05 else 'Moderate' if ammonia_value <= 0.5 else 'Poor'

    steps = []
    for category, info in ranges.items():
        color = info['color']
        if category == current_category:
            steps.append({
                'range': info['range'],
                'color': color,
                'line': {'color': "#FFD700", 'width': 3},
                'thickness': 0.8
            })
        else:
            steps.append({
                'range': info['range'],
                'color': color,
                'line': {'color': "#FFD700", 'width': 1},
                'thickness': 0.6
            })

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ammonia_value,
        title={'text': "Ammonia (mg/L)"},
        gauge={
            'axis': {'range': [None, 1.0]},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_nitrate_gauge(nitrate_value):
    """Create gauge for nitrate levels"""
    ranges = {
        'Safe': {'range': [0, 10], 'color': '#008000'},
        'Elevated': {'range': [10, 20], 'color': '#FFD700'},
        'High': {'range': [20, 40], 'color': '#FF4444'}
    }

    current_category = 'Safe' if nitrate_value <= 10 else 'Elevated' if nitrate_value <= 20 else 'High'

    steps = []
    for category, info in ranges.items():
        color = info['color']
        if category == current_category:
            steps.append({
                'range': info['range'],
                'color': color,
                'line': {'color': "#008000", 'width': 3},
                'thickness': 0.8
            })
        else:
            steps.append({
                'range': info['range'],
                'color': color,
                'line': {'color': "#008000", 'width': 1},
                'thickness': 0.6
            })

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=nitrate_value,
        title={'text': "Nitrate (mg/L)"},
        gauge={
            'axis': {'range': [None, 40]},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_phosphate_gauge(pho_value):
    """Create gauge for phosphate levels"""
    ranges = {
        'Low': {'range': [0, 2.0], 'color': '#FFD700'},
        'Normal': {'range': [2.0, 4.5], 'color': '#008000'},
        'High': {'range': [4.5, 6.0], 'color': '#FF4444'}
    }

    current_category = 'Low' if pho_value < 2.0 else 'Normal' if pho_value <= 4.5 else 'High'

    steps = []
    for category, info in ranges.items():
        color = info['color']
        if category == current_category:
            steps.append({
                'range': info['range'],
                'color': color,
                'line': {'color': "#008000", 'width': 3},
                'thickness': 0.8
            })
        else:
            steps.append({
                'range': info['range'],
                'color': color,
                'line': {'color': "#008000", 'width': 1},
                'thickness': 0.6
            })

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pho_value,
        title={'text': "Phosphate (mg/L)"},
        gauge={
            'axis': {'range': [None, 6.0]},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 4.5
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def display_wqi_section(wqi_value):
    """Display WQI gauge section in Streamlit"""
    st.subheader("📊 Water Quality Index (WQI)")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = create_wqi_gauge(wqi_value)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### WQI Status")

        if wqi_value <= 25:
            st.error("🔴 **Poor Water Quality**")
            st.write("Immediate action required. Water may be harmful for aquatic life and human use.")
            status_color = "#FF4444"
        elif wqi_value <= 50:
            st.warning("🟠 **Fair Water Quality**")
            st.write("Water quality is below acceptable standards. Monitoring and treatment recommended.")
            status_color = "#FF8C00"
        elif wqi_value <= 70:
            st.info("🟡 **Average Water Quality**")
            st.write("Water quality is acceptable but could be improved. Regular monitoring advised.")
            status_color = "#FFD700"
        elif wqi_value <= 90:
            st.success("🟢 **Good Water Quality**")
            st.write("Water quality is good and marine life can thrive. Maintain current management practices.")
            status_color = "#90EE90"
        else:
            st.success("⭐ **Excellent Water Quality**")
            st.write("Outstanding water quality! This water body is in excellent condition.")
            status_color = "#32CD32"

        st.markdown(
            f"""
            <div style="
                background-color: {status_color}; 
                color: white; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center;
                font-weight: bold;
                font-size: 18px;
                margin: 10px 0;
                border: 2px solid #333;
            ">
                Current WQI<br>
                <span style="font-size: 24px;">{wqi_value:.1f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


# === WQI Calculation ===
def calculate_wqi(data_row):
    weights = {
        'ph': 0.11, 'temperature': 0.1, 'dissolved_oxygen': 0.17,
        'ammonia': 0.17, 'nitrate': 0.22, 'pho': 0.23
    }
    sub_index = {
        'ph': lambda v: max(0, 100 - abs(7 - v) * 15),
        'temperature': lambda v: max(0, 100 - abs(25 - v) * 3),
        'dissolved_oxygen': lambda v: min(v / 8.0 * 100, 100),
        'ammonia': lambda v: 100 if v <= 0.05 else 50 if v <= 0.5 else 20,
        'nitrate': lambda v: 100 if v <= 10 else 50 if v <= 20 else 20,
        'pho': lambda v: 100 if (2.0 <= v <= 4.5) else 50 if (1.0 <= v < 2.0 or 4.5 < v <= 5.0) else 20
    }
    return round(sum(sub_index[k](data_row[k]) * weights[k] for k in weights if k in data_row), 2)


# === Pollutant Classification ===
def classify_pollutants(data_row):
    return {
        'ammonia_class': "Good" if data_row['ammonia'] <= 0.05 else "Moderate" if data_row[
                                                                                      'ammonia'] <= 0.5 else "Poor",
        'nitrate_class': "Safe" if data_row['nitrate'] <= 10 else "Elevated" if data_row['nitrate'] <= 20 else "High",
        'pho_class': "Normal" if (2.0 <= data_row['pho'] <= 4.5) else "High" if data_row['pho'] > 4.5 else "Low"
    }


# === Streamlit UI ===
st.set_page_config(page_title="📅 Taal Lake Water Quality Data Analysis", layout="wide")
st.title("📅 Taal Lake Water Quality Data Analysis")

# === Load CSV Data ===
df = load_csv_data()

if df is not None:
    # === SIDEBAR CONTROLS ===
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")

        # Get unique locations from CSV
        locations = df['location'].unique().tolist()  # Adjust column name as needed

        # Location Selection
        st.subheader("Per Location")
        location = st.selectbox(
            "Select Location:",
            options=locations,
            index=0,
            help="Choose the water body location for analysis"
        )

        # Time Period Selection
        st.subheader("📅 Time Period")
        time_period_type = st.selectbox(
            "Select Time Period Type:",
            options=["Monthly", "Yearly"],
            index=0,
            help="Choose whether to view data monthly or yearly (2013-2025)"
        )

        if time_period_type == "Yearly":
            # Automatic yearly view from 2013 to 2025
            start_date = pd.Timestamp("2013-01-01")
            end_date = pd.Timestamp("2025-12-31")
            period_display = "2013-2025 (All Years)"

        else:  # Monthly
            # Automatic monthly view from 2013 to 2025
            start_date = pd.Timestamp("2013-01-01")
            end_date = pd.Timestamp("2025-12-31")
            period_display = "2013-2025 (All Months)"

        # Input Variable Selection
        st.subheader("Input Variables")

        # Define parameter groups based on your CSV columns
        # Adjust these lists based on your actual CSV column names
        water_quality_params = ['ph', 'temperature', 'dissolved_oxygen', 'ammonia', 'nitrate', 'pho']
        environmental_params = ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION']
        volcanic_params = ['co2', 'so2']

        # Filter parameters that actually exist in the CSV
        available_water_params = [p for p in water_quality_params if p in df.columns]
        available_env_params = [p for p in environmental_params if p in df.columns]
        available_volcanic_params = [p for p in volcanic_params if p in df.columns]

        input_variable_type = st.selectbox(
            "Select Parameter Display Type:",
            options=[
                "Water Parameters",
                "Water Parameters + Meteorological Data",
                "Water Parameters + Volcanic Activity Data",
                "All Three Parameters"
            ],
            index=0,
            help="Choose which parameters to display in the dashboard"
        )

        # Map selection to actual parameters
        if input_variable_type == "Water Parameters":
            selected_params = available_water_params
        elif input_variable_type == "Water Parameters + Meteorological Data":
            selected_params = available_water_params + available_env_params
        elif input_variable_type == "Water Parameters + Volcanic Activity Data":
            selected_params = available_water_params + available_volcanic_params
        else:  # All Three Parameters
            selected_params = available_water_params + available_env_params + available_volcanic_params

        st.info(f"📊 **Current Selection:** {input_variable_type}")

        # Category Selection
        st.subheader("📊 Category Analysis")
        category_options = [
            "Current (Latest)", "1 Month", "2 Months", "3 Months", "4 Months", "5 Months", "6 Months",
            "7 Months", "8 Months", "9 Months", "10 Months", "11 Months", "1 Year",
            "2 Years", "3 Years"
        ]

        selected_category = st.selectbox(
            "Select Time Range for WQI & Pollutant Analysis:",
            options=category_options,
            index=0,
            help="Choose the time period to calculate average WQI and pollutant levels"
        )

    # === Filter Data ===
    # Filter by location
    filtered_data = df[df['location'] == location].copy()

    # Filter by the calculated date range
    filtered_data = filtered_data[
        (filtered_data['date'] >= start_date) &
        (filtered_data['date'] <= end_date)
        ]

    # Store original filtered data for WQI and pollutant analysis
    original_filtered_data = filtered_data.copy()

    # Aggregate data ONLY for line charts based on time period selection
    if time_period_type == "Yearly":
        # Group by year and calculate mean for LINE CHARTS ONLY
        filtered_data['year'] = filtered_data['date'].dt.year

        # Get numeric columns for aggregation
        numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
        # Remove non-parameter columns if they exist
        exclude_cols = ['year']
        numeric_columns = [col for col in numeric_columns if col not in exclude_cols]

        # Aggregate by year
        yearly_data = filtered_data.groupby('year')[numeric_columns].mean().reset_index()
        # Create a date column for the year (use January 1st of each year)
        yearly_data['date'] = pd.to_datetime(yearly_data['year'].astype(str) + '-01-01')

        # Replace filtered_data with aggregated data FOR LINE CHARTS
        chart_data = yearly_data.copy()

    else:  # Monthly
        # Group by year-month and calculate mean for LINE CHARTS ONLY
        filtered_data['year_month'] = filtered_data['date'].dt.to_period('M')

        # Get numeric columns for aggregation
        numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()

        # Aggregate by month
        monthly_data = filtered_data.groupby('year_month')[numeric_columns].mean().reset_index()
        # Convert period back to datetime (use first day of each month)
        monthly_data['date'] = monthly_data['year_month'].dt.to_timestamp()

        # Replace filtered_data with aggregated data FOR LINE CHARTS
        chart_data = monthly_data.copy()

    # Calculate WQI for original (non-aggregated) data
    if all(param in original_filtered_data.columns for param in
           ['ph', 'temperature', 'dissolved_oxygen', 'ammonia', 'nitrate', 'pho']):
        original_filtered_data['WQI'] = original_filtered_data.apply(lambda row: calculate_wqi(row), axis=1)

    # Also calculate WQI for chart data if needed
    if all(param in chart_data.columns for param in
           ['ph', 'temperature', 'dissolved_oxygen', 'ammonia', 'nitrate', 'pho']):
        chart_data['WQI'] = chart_data.apply(lambda row: calculate_wqi(row), axis=1)

    # === Display Results ===
    if not filtered_data.empty and selected_params:
        st.success(
            f"📍 **Location:** {location} | 📅 **Period:** {period_display} | 🔢 **Data Points:** {len(filtered_data)} | 🔢 **Parameters:** {len(selected_params)}")

        # Calculate WQI for each row (if water quality parameters are available)
        if all(param in filtered_data.columns for param in
               ['ph', 'temperature', 'dissolved_oxygen', 'ammonia', 'nitrate', 'pho']):
            filtered_data['WQI'] = filtered_data.apply(lambda row: calculate_wqi(row), axis=1)

        # === Display Parameter Trends ===
        st.subheader("📈 Parameter Trends Over Time (2013-2025)")

        # Water Quality Parameters
        water_params_to_show = [p for p in selected_params if p in available_water_params]
        if water_params_to_show:
            st.subheader("🌊 Water Quality Parameters")

            water_cols = st.columns(2)
            for i, param in enumerate(water_params_to_show):
                with water_cols[i % 2]:
                    units = {
                        'ph': 'pH units',
                        'temperature': '°C',
                        'dissolved_oxygen': 'mg/L',
                        'ammonia': 'mg/L',
                        'nitrate': 'mg/L',
                        'pho': 'mg/L'
                    }
                    unit_str = units.get(param, '')
                    st.caption(f"📊 {param.replace('_', ' ').title()} {unit_str}")

                    # Create line chart using chart_data (aggregated data)
                    chart_series = chart_data.set_index('date')[param].dropna()
                    if not chart_series.empty:
                        st.line_chart(chart_series, height=300, use_container_width=True)
                    else:
                        st.warning(f"No data available for {param}")

        # Environmental Parameters
        env_params_to_show = [p for p in selected_params if p in available_env_params]
        if env_params_to_show:
            st.subheader("🌤️ Meteorological Parameters")

            env_cols = st.columns(2)
            for i, param in enumerate(env_params_to_show):
                with env_cols[i % 2]:
                    st.caption(f"🌡️ {param.replace('_', ' ').title()}")

                    # Use chart_data for line charts
                    chart_series = chart_data.set_index('date')[param].dropna()
                    if not chart_series.empty:
                        st.line_chart(chart_series, height=300, use_container_width=True)
                    else:
                        st.warning(f"No data available for {param}")

        # Volcanic Parameters
        volcanic_params_to_show = [p for p in selected_params if p in available_volcanic_params]
        if volcanic_params_to_show:
            st.subheader("🌋 Volcanic Activity Parameters")

            volcanic_cols = st.columns(2)
            for i, param in enumerate(volcanic_params_to_show):
                with volcanic_cols[i % 2]:
                    st.caption(f"🌋 {param.upper()}")

                    # Use chart_data for line charts
                    chart_series = chart_data.set_index('date')[param].dropna()
                    if not chart_series.empty:
                        st.line_chart(chart_series, height=300, use_container_width=True)
                    else:
                        st.warning(f"No data available for {param}")

        # === WQI Section ===
        if 'WQI' in original_filtered_data.columns:
            # Determine which data to use for WQI and pollutant analysis
            if selected_category == "Current (Latest)":
                # Use latest single data point from ORIGINAL data
                analysis_data = original_filtered_data.iloc[-1]
                period_description = "Current (Latest Reading)"
            else:
                # Convert selection to months and calculate averages from ORIGINAL data
                category_months_map = {
                    "1 Month": 1, "2 Months": 2, "3 Months": 3, "4 Months": 4, "5 Months": 5, "6 Months": 6,
                    "7 Months": 7, "8 Months": 8, "9 Months": 9, "10 Months": 10, "11 Months": 11,
                    "1 Year": 12, "2 Years": 24, "3 Years": 36
                }
                selected_months = category_months_map[selected_category]

                # Calculate date range for category analysis
                latest_date = original_filtered_data['date'].max()
                category_start_date = latest_date - pd.DateOffset(months=selected_months)

                # Filter ORIGINAL data for category period
                category_data = original_filtered_data[
                    (original_filtered_data['date'] >= category_start_date) &
                    (original_filtered_data['date'] <= latest_date)
                    ].copy()

                if not category_data.empty:
                    # Calculate averages for all parameters
                    analysis_data = category_data[
                        ['ph', 'temperature', 'dissolved_oxygen', 'ammonia', 'nitrate', 'pho']].mean()
                    period_description = f"Average over last {selected_category} ({len(category_data)} data points)"
                else:
                    st.warning(f"⚠️ No data available for the last {selected_category}")
                    analysis_data = original_filtered_data.iloc[-1]  # Fallback to latest
                    period_description = "Current (Latest Reading - Fallback)"

            # Calculate WQI based on the analysis data
            current_wqi = calculate_wqi(analysis_data)

            # Display period info
            st.info(f"📊 **Analysis Period:** {period_description}")

            # === WQI Section ===
            display_wqi_section(current_wqi)

            # === Pollutant Levels ===
            if all(param in analysis_data.index for param in ['ammonia', 'nitrate', 'pho']):
                st.subheader("🧪 Pollutant Levels")

                # Get values (either current or averaged)
                current_ammonia = analysis_data['ammonia']
                current_nitrate = analysis_data['nitrate']
                current_pho = analysis_data['pho']

                # Get classifications
                classifications = classify_pollutants(analysis_data)

                # Create gauge columns
                class_col1, class_col2, class_col3 = st.columns(3)

                with class_col1:
                    ammonia_fig = create_ammonia_gauge(current_ammonia)
                    st.plotly_chart(ammonia_fig, use_container_width=True)

                    st.metric(
                        label="Value" if selected_category == "Current (Latest)" else "Average Value",
                        value=f"{current_ammonia:.3f} mg/L",
                        help=f"Classification: {classifications['ammonia_class']}"
                    )

                    if classifications['ammonia_class'] == 'Good':
                        st.success(f"✅ {classifications['ammonia_class']}")
                    elif classifications['ammonia_class'] == 'Moderate':
                        st.warning(f"⚠️ {classifications['ammonia_class']}")
                    else:
                        st.error(f"❌ {classifications['ammonia_class']}")

                with class_col2:
                    nitrate_fig = create_nitrate_gauge(current_nitrate)
                    st.plotly_chart(nitrate_fig, use_container_width=True)

                    st.metric(
                        label="Value" if selected_category == "Current (Latest)" else "Average Value",
                        value=f"{current_nitrate:.2f} mg/L",
                        help=f"Classification: {classifications['nitrate_class']}"
                    )

                    if classifications['nitrate_class'] == 'Safe':
                        st.success(f"✅ {classifications['nitrate_class']}")
                    elif classifications['nitrate_class'] == 'Elevated':
                        st.warning(f"⚠️ {classifications['nitrate_class']}")
                    else:
                        st.error(f"❌ {classifications['nitrate_class']}")

                with class_col3:
                    phosphate_fig = create_phosphate_gauge(current_pho)
                    st.plotly_chart(phosphate_fig, use_container_width=True)

                    st.metric(
                        label="Value" if selected_category == "Current (Latest)" else "Average Value",
                        value=f"{current_pho:.2f} mg/L",
                        help=f"Classification: {classifications['pho_class']}"
                    )

                    if classifications['pho_class'] == 'Normal':
                        st.success(f"✅ {classifications['pho_class']}")
                    else:
                        st.warning(f"⚠️ {classifications['pho_class']}")

        # === Data Table ===
        with st.expander("📋 View Raw Data", expanded=False):
            display_columns = ['date'] + selected_params
            if 'WQI' in original_filtered_data.columns:
                display_columns.append('WQI')

            available_display_columns = [col for col in display_columns if col in original_filtered_data.columns]
            st.dataframe(
                original_filtered_data[available_display_columns].sort_values('date', ascending=False),
                use_container_width=True
            )

    else:
        if filtered_data.empty:
            st.warning("⚠️ No data available for the selected location and date range.")
        else:
            st.warning("⚠️ Please select at least one parameter type to display results.")

else:
    st.error("❌ Unable to load CSV data. Please check the file path and format.")

# === General Threshold Information ===
st.subheader("⚠️ Water Quality Threshold Guidelines")
st.markdown("""
**When parameters exceed these thresholds, consider taking action:**

1. **Ammonia (NH₃)**
   - Good: ≤ 0.05 mg/L
   - Moderate: ≤ 0.5 mg/L
   - Poor: > 0.5 mg/L

2. **Nitrate (NO₃)**
   - Safe: ≤ 10 mg/L
   - Elevated: ≤ 20 mg/L
   - High: > 20 mg/L

3. **Phosphate (PO₄)**
   - Normal: 2.0-4.5 mg/L
   - High: > 4.5 mg/L
   - Low: < 2.0 mg/L

4. **pH**
   - Ideal range: 6.5-8.5

5. **Dissolved Oxygen**
   - Healthy: > 5 mg/L

6. **Temperature**
   - Optimal: 20-25°C
""")

if 'WQI' in original_filtered_data.columns and all(param in analysis_data.index for param in ['ammonia', 'nitrate', 'pho']):
            display_insights_section(current_wqi, analysis_data, classifications, location, period_description)
