# 🌊 Taal Lake Water Quality Dashboard

A comprehensive Streamlit dashboard for analyzing water quality data from Taal Lake, Philippines. This application provides real-time water quality monitoring, trend analysis, and actionable insights for lake management.

## 🚀 Features

- **Water Quality Index (WQI) Calculation**: Real-time WQI calculation with visual gauges
- **Parameter Monitoring**: Track ammonia, nitrate, phosphate, pH, dissolved oxygen, and temperature
- **Time Series Analysis**: View trends over monthly and yearly periods (2013-2025)
- **Pollutant Classification**: Automatic classification of pollutant levels with color-coded alerts
- **Location-based Analysis**: Multi-location support for different areas of Taal Lake
- **Insights & Recommendations**: AI-generated insights and management recommendations
- **Interactive Visualizations**: Dynamic charts and gauges using Plotly

## 📊 Dashboard Sections

1. **Parameter Trends**: Line charts showing water quality parameter changes over time
2. **WQI Gauge**: Speedometer-style gauge showing current water quality index
3. **Pollutant Levels**: Individual gauges for ammonia, nitrate, and phosphate levels
4. **Insights & Recommendations**: Comprehensive analysis with actionable recommendations
5. **Raw Data View**: Expandable table showing all collected data

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/taal-lake-dashboard.git
   cd taal-lake-dashboard
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 File Structure

```
taal-lake-dashboard/
├── main.py                      # Main Streamlit application
├── water_quality_data.csv       # Water quality dataset
├── hybrid_model_v2.keras        # Trained ML model (optional)
├── input_scaler_std.save        # Input data scaler (optional)
├── target_scaler_std.save       # Target data scaler (optional)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎛️ Usage

### Dashboard Controls
- **Location Selection**: Choose from different monitoring locations around Taal Lake
- **Time Period**: Select between monthly and yearly data aggregation
- **Parameter Display**: Choose which parameter groups to display:
  - Water Parameters only
  - Water + Meteorological Data
  - Water + Volcanic Activity Data
  - All Parameters
- **Analysis Period**: Select time range for WQI and pollutant analysis

### Water Quality Parameters
- **pH**: Acidity/alkalinity levels
- **Temperature**: Water temperature in Celsius
- **Dissolved Oxygen**: Available oxygen for aquatic life
- **Ammonia**: Toxic nitrogen compound levels
- **Nitrate**: Nutrient levels that can cause eutrophication
- **Phosphate**: Another key nutrient affecting water quality

### Environmental Parameters
- **Rainfall**: Precipitation data
- **Temperature**: Air temperature (TMAX, TMIN)
- **Humidity**: Relative humidity (RH)
- **Wind**: Speed and direction

### Volcanic Activity Parameters
- **CO2**: Carbon dioxide emissions
- **SO2**: Sulfur dioxide emissions

## 📈 Water Quality Index (WQI) Scale

- **Excellent (91-100)**: Outstanding water quality
- **Good (71-90)**: Suitable for all uses
- **Average (51-70)**: Acceptable but could be improved
- **Fair (26-50)**: Below standards, treatment recommended
- **Poor (0-25)**: Immediate action required

## 🧪 Pollutant Thresholds

### Ammonia (NH₃)
- **Good**: ≤ 0.05 mg/L
- **Moderate**: ≤ 0.5 mg/L
- **Poor**: > 0.5 mg/L

### Nitrate (NO₃)
- **Safe**: ≤ 10 mg/L
- **Elevated**: ≤ 20 mg/L
- **High**: > 20 mg/L

### Phosphate (PO₄)
- **Normal**: 2.0-4.5 mg/L
- **High**: > 4.5 mg/L
- **Low**: < 2.0 mg/L

## 🌋 About Taal Lake

Taal Lake is a freshwater crater lake located in Batangas Province, Philippines. It is situated within the Taal Caldera and contains Taal Volcano Island. The lake is a critical ecosystem that supports local communities through fishing and tourism while requiring careful environmental management due to its volcanic nature.

## 🔬 Data Sources

The dashboard uses water quality data collected from various monitoring stations around Taal Lake, including:
- Chemical parameters (pH, nutrients, pollutants)
- Physical parameters (temperature, dissolved oxygen)
- Meteorological data
- Volcanic activity indicators

## 🤝 Contributing

We welcome contributions to improve the dashboard! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or support, please contact:
- **Project Maintainer**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [yourusername]

## 🙏 Acknowledgments

- Taal Lake monitoring stations and data collectors
- Environmental agencies and research institutions
- Open-source community for tools and libraries used

---

**Note**: This dashboard is designed for educational and research purposes. For critical environmental decisions, please consult with qualified environmental professionals and official monitoring agencies.
