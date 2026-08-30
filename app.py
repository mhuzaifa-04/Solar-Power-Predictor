import io
import requests
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ReportLab imports for PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# 1. Page Configuration
st.set_page_config(
    page_title="Solar AI Pro | Yield & Financial Forecasting",
    page_icon="☀️",
    layout="wide"
)

# 2. Advanced Glassmorphism CSS Injector
def apply_custom_ui():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0b0f19 100%) !important;
        background-attachment: fixed;
    }

    /* Headings & Text Clarity */
    h1, h2, h3, h4, span, label, p {
        color: #f8fafc !important;
    }
    .stCaption {
        color: #94a3b8 !important;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border 0.2s ease;
    }
    .glass-card:hover {
        border: 1px solid rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-2px);
    }

    /* Fix Streamlit Navigation Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        background-color: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px !important;
    }
    button[aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #6366f1 !important;
    }

    /* Input Field Styling */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        transform: translateY(-1px);
    }

    /* Sidebar Clean styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

apply_custom_ui()

# KPI Card Helper Function
def render_kpi_card(title, value, subtext="", status_color="#6366f1"):
    st.markdown(f"""
    <div class="glass-card">
        <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.4rem;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {status_color}; margin-bottom: 0.4rem;">{value}</div>
        <div style="color: #64748b; font-size: 0.8rem; font-weight: 400;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

# 3. Load Trained Model
@st.cache_resource
def load_trained_model():
    return joblib.load('models/solar_xgboost_model.pkl')

try:
    model = load_trained_model()
except Exception:
    st.error("❌ Model missing! Ensure 'python 2_train_model.py' has been run.")
    st.stop()

# Helper Function: Fetch Real-Time Weather via Open-Meteo API
def fetch_city_weather(city_name):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get('results'):
            return None, f"City '{city_name}' not found."
        
        lat = geo_res['results'][0]['latitude']
        lon = geo_res['results'][0]['longitude']
        country = geo_res['results'][0].get('country', '')

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,direct_normal_irradiance,cloud_cover"
        w_res = requests.get(weather_url).json()
        
        current = w_res.get('current', {})
        temp = current.get('temperature_2m', 25.0)
        irradiance_raw = current.get('direct_normal_irradiance', 500.0)
        irradiance = min(1.2, max(0.0, irradiance_raw / 800.0))

        return {
            'city': city_name.capitalize(),
            'country': country,
            'temp': temp,
            'irradiance': irradiance
        }, None
    except Exception as e:
        return None, f"API Error: {str(e)}"

# Helper Function: Generate PDF Client Report
def generate_pdf_report(client_name, kw_rating, location, predicted_power, daily_kwh, annual_savings, payback_year, total_25yr_net, currency):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1E222D'),
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#555555'),
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#6366f1'),
        spaceBefore=12,
        spaceAfter=8
    )

    story.append(Paragraph("☀️ Solar AI Feasibility & Audit Proposal", title_style))
    story.append(Paragraph(f"Prepared for: <b>{client_name}</b> | System Location: <b>{location}</b>", subtitle_style))
    story.append(Spacer(1, 10))

    summary_data = [
        ["Parameter", "Forecasted Value"],
        ["Target System Rating", f"{kw_rating:.2f} kWp"],
        ["Predicted Peak AC Power", f"{predicted_power:.2f} kW"],
        ["Est. Daily Energy Yield", f"{daily_kwh:.2f} kWh/day"],
        ["Projected Annual Utility Savings", f"{currency}{annual_savings:,.2f}"],
        ["Estimated Break-Even Period", f"{payback_year if payback_year else '25+'} Years"],
        ["Cumulative 25-Year Net Profit", f"{currency}{total_25yr_net:,.2f}"]
    ]

    t = Table(summary_data, colWidths=[240, 240])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1E222D')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    story.append(Paragraph("System Guarantee & Model Disclaimer", heading_style))
    disclaimer_text = ("This proposal utilizes XGBoost machine learning algorithms trained on site-specific solar irradiance "
                       "and meteorological observations. Actual real-world yield may vary slightly based on local weather deviations, panel orientation, and maintenance.")
    story.append(Paragraph(disclaimer_text, styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# 4. App Title Header
st.title("☀️ Solar AI Intelligence & Financial Forecasting")
st.caption("Powered by XGBoost Machine Learning Engine")

# 5. Sidebar Controls & Developer Signature
with st.sidebar:
    st.header("🎛️ Input Mode")
    input_mode = st.radio("Choose Weather Input Source:", ["Manual Sliders", "🌐 Live City Weather API"])
    
    if input_mode == "Manual Sliders":
        st.subheader("⚙️ Manual Weather Controls")
        irradiance = st.slider("Solar Irradiance (W/m²)", 0.0, 1.2, 0.75, step=0.05)
        amb_temp = st.slider("Ambient Temp (°C)", 10.0, 50.0, 30.0, step=0.5)
        mod_temp = st.slider("Panel/Module Temp (°C)", 10.0, 70.0, 42.0, step=0.5)
        hour = st.slider("Hour of Day (24h)", 0, 23, 13)
        month = st.selectbox("Month of Year", list(range(1, 13)), index=5)
        location_name = "Custom Manual Site"
    else:
        st.subheader("🌐 Live City Search")
        city_input = st.text_input("Enter City Name:", value="Nagpur")
        hour = st.slider("Hour of Day (24h)", 0, 23, 13)
        month = st.selectbox("Month of Year", list(range(1, 13)), index=5)
        
        weather_data, err = fetch_city_weather(city_input)
        if weather_data:
            st.success(f"📍 Location: {weather_data['city']}, {weather_data['country']}")
            st.info(f"🌡️ Temp: {weather_data['temp']}°C | ☀️ Irradiance: {weather_data['irradiance']:.2f} W/m²")
            amb_temp = weather_data['temp']
            irradiance = weather_data['irradiance']
            mod_temp = amb_temp + (irradiance * 15.0)
            location_name = f"{weather_data['city']}, {weather_data['country']}"
        else:
            st.warning(err or "Unable to fetch live weather. Defaulting to standard values.")
            amb_temp = 28.0
            irradiance = 0.65
            mod_temp = 38.0
            location_name = "Default Site"

    st.markdown("---")
    st.header("💰 Tariff & Financial Settings")
    currency = st.selectbox("Currency Unit", ["$", "₹", "€", "£"], index=1)
    tariff_rate = st.number_input(f"Utility Tariff ({currency}/kWh)", min_value=0.01, value=8.0 if currency=="₹" else 0.15, step=0.5)

    # --- AUTHOR & SOCIAL LINKS ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 500;">
            Developed by <span style="color: #6366f1; font-weight: 700;">Mohammad Huzaifa</span> ✨
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; margin-top: 0.6rem;">
            <a href="https://www.linkedin.com/in/mhuzaifa04" target="_blank" style="text-decoration: none;">
                <span style="background: rgba(99, 102, 241, 0.15); color: #6366f1; border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 8px; padding: 6px 12px; font-size: 0.8rem; font-weight: 600;">
                    💼 LinkedIn
                </span>
            </a>
            <a href="https://github.com/mhuzaifa-04" target="_blank" style="text-decoration: none;">
                <span style="background: rgba(255, 255, 255, 0.08); color: #f8fafc; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px; padding: 6px 12px; font-size: 0.8rem; font-weight: 600;">
                    🐙 GitHub
                </span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 6. Model Prediction & Computations
temp_diff = mod_temp - amb_temp
input_features = pd.DataFrame({
    'IRRADIANCE': [irradiance],
    'AMBIENT_TEMPERATURE': [amb_temp],
    'MODULE_TEMPERATURE': [mod_temp],
    'TEMP_DIFF': [temp_diff],
    'HOUR': [hour],
    'MONTH': [month]
})

predicted_power = max(0.0, float(model.predict(input_features)[0]))
estimated_daily_kwh = (predicted_power / 1000.0) * 6.0  
daily_savings = estimated_daily_kwh * tariff_rate
annual_savings = daily_savings * 365.0

# Pre-calculate 25-Year financial data
system_cost_default = 250000 if currency == "₹" else 5000
degradation_rate_default = 0.005
years = np.arange(1, 26)
cumulative_cashflow = []
current_cashflow = -system_cost_default
payback_year = None

for y in years:
    degraded_savings = annual_savings * ((1.0 - degradation_rate_default) ** (y - 1))
    current_cashflow += degraded_savings
    cumulative_cashflow.append(current_cashflow)
    if current_cashflow >= 0 and payback_year is None:
        payback_year = y

total_25yr_net = cumulative_cashflow[-1]

# 7. Unpack Dashboard Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Performance Dashboard", 
    "🧠 ML Feature Insights", 
    "📁 Batch Prediction",
    "💵 25-Year ROI & Payback",
    "🔋 Battery & Net-Metering",
    "📄 Client PDF Export"
])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card("Predicted AC Power", f"{predicted_power:.2f} kW", "Peak Output Generation", "#FFB800")
    with col2:
        render_kpi_card("Est. Daily Savings", f"{currency}{daily_savings:.2f}", "Based on local tariff", "#10b981")
    with col3:
        render_kpi_card("Projected Annual Savings", f"{currency}{annual_savings:,.2f}", "365-day cumulative", "#6366f1")

    st.write("")
    st.subheader("📈 24-Hour Simulated Power Profile")
    hours = np.arange(0, 24)
    hourly_curve = [
        max(0.0, np.sin((h - 6) / 12.0 * np.pi) * predicted_power) if 6 <= h <= 18 else 0.0 
        for h in hours
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=hourly_curve, mode='lines+markers', name='AC Power Output',
        fill='tozeroy', fillcolor='rgba(255, 184, 0, 0.2)', line=dict(color='#FFB800', width=3)
    ))
    fig.update_layout(
        xaxis=dict(title="Hour of Day", tickmode='linear', tick0=0, dtick=2),
        yaxis=dict(title="AC Power Output (kW)"),
        template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🌲 Feature Importance Analysis")
    importances = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': input_features.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    feat_fig = go.Figure(go.Bar(
        x=features_df['Importance'], y=features_df['Feature'], orientation='h', marker=dict(color='#6366f1')
    ))
    feat_fig.update_layout(
        template="plotly_dark", height=350, xaxis_title="Relative Feature Importance Weight",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(feat_fig, use_container_width=True)

with tab3:
    st.subheader("📁 Upload CSV for Bulk Solar Forecasting")
    uploaded_file = st.file_uploader("Upload weather log (.csv)", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = ['IRRADIANCE', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'HOUR', 'MONTH']
        if all(col in batch_df.columns for col in required_cols):
            batch_df['TEMP_DIFF'] = batch_df['MODULE_TEMPERATURE'] - batch_df['AMBIENT_TEMPERATURE']
            batch_df['PREDICTED_POWER_KW'] = model.predict(batch_df[input_features.columns]).clip(min=0)
            st.success(f"✅ Predicted power for {len(batch_df)} timestamp records!")
            st.dataframe(batch_df.head(10))
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Predictions CSV", data=csv_data, file_name="solar_power_predictions.csv", mime="text/csv")
        else:
            st.error(f"CSV must contain required columns: {required_cols}")

with tab4:
    st.subheader("💵 Long-Term Financial ROI & System Payback Calculator")
    col_roi1, col_roi2 = st.columns(2)
    with col_roi1:
        system_cost = st.number_input(f"Total System Installation Cost ({currency})", min_value=1000, value=system_cost_default, step=1000)
    with col_roi2:
        degradation_rate = st.slider("Annual Panel Degradation Rate (%)", 0.1, 2.0, 0.5, step=0.1) / 100.0

    yearly_cashflow = []
    current_cashflow = -system_cost
    payback_year_dynamic = None

    for y in years:
        degraded_savings = annual_savings * ((1.0 - degradation_rate) ** (y - 1))
        current_cashflow += degraded_savings
        yearly_cashflow.append(current_cashflow)
        if current_cashflow >= 0 and payback_year_dynamic is None:
            payback_year_dynamic = y

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        render_kpi_card("Break-Even Period", f"{payback_year_dynamic} Years" if payback_year_dynamic else "25+ Years", "Estimated payback time", "#00E5FF")
    with col_kpi2:
        render_kpi_card("Net 25-Year Profit", f"{currency}{yearly_cashflow[-1]:,.2f}", "Lifetime net return", "#10b981")
    with col_kpi3:
        roi_percentage = (yearly_cashflow[-1] / system_cost) * 100.0
        render_kpi_card("25-Year ROI", f"{roi_percentage:.1f}%", "Return on investment", "#6366f1")

    roi_fig = go.Figure()
    roi_fig.add_trace(go.Scatter(x=years, y=yearly_cashflow, mode='lines+markers', name='Cumulative Cash Flow', line=dict(color='#00E5FF', width=3)))
    roi_fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-Even Point ($0)")
    roi_fig.update_layout(
        title="25-Year Cumulative Financial Cash Flow", xaxis=dict(title="Year of Operation", tickmode='linear', dtick=2),
        yaxis=dict(title=f"Net Cumulative Cash Flow ({currency})"), template="plotly_dark", height=380,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(roi_fig, use_container_width=True)

with tab5:
    st.subheader("🔋 Battery Energy Storage & Grid Net-Metering Simulation")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=0.0, value=13.5, step=1.0)
    with col_b2:
        avg_hourly_demand = st.number_input("Avg Household Demand (kW)", min_value=0.1, value=1.5, step=0.1)
    with col_b3:
        export_tariff = st.number_input(f"Grid Export Feed-in Tariff ({currency}/kWh)", min_value=0.0, value=tariff_rate * 0.6, step=0.5)

    battery_soc = 0.0
    efficiency = 0.89
    solar_gen, household_demand, battery_charge_state, grid_export, grid_import = [], [], [], [], []

    for h in range(24):
        demand = avg_hourly_demand * (2.2 if 18 <= h <= 22 else 0.8)
        household_demand.append(demand)
        gen = max(0.0, np.sin((h - 6) / 12.0 * np.pi) * (predicted_power / 100.0)) if 6 <= h <= 18 else 0.0
        solar_gen.append(gen)
        net_energy = gen - demand
        
        if net_energy > 0:
            charge_possible = min(net_energy * efficiency, battery_capacity - battery_soc)
            battery_soc += charge_possible
            grid_export.append(max(0.0, net_energy - (charge_possible / efficiency)))
            grid_import.append(0.0)
        else:
            deficit = abs(net_energy)
            discharge_possible = min(deficit, battery_soc)
            battery_soc -= discharge_possible
            grid_import.append(deficit - discharge_possible)
            grid_export.append(0.0)
            
        battery_charge_state.append(battery_soc)

    total_exported = sum(grid_export)
    total_imported = sum(grid_import)
    daily_net_bill = (total_imported * tariff_rate) - (total_exported * export_tariff)

    col_mb1, col_mb2, col_mb3 = st.columns(3)
    with col_mb1:
        render_kpi_card("Daily Grid Export", f"{total_exported:.2f} kWh", "Surplus feed-in energy", "#10b981")
    with col_mb2:
        render_kpi_card("Daily Grid Import", f"{total_imported:.2f} kWh", "Utility grid draw", "#ef4444")
    with col_mb3:
        status_color = "#10b981" if daily_net_bill <= 0 else "#ef4444"
        render_kpi_card("Net Daily Utility Bill", f"{currency}{daily_net_bill:.2f}", "Net cost after export credit", status_color)

    bat_fig = go.Figure()
    bat_fig.add_trace(go.Scatter(x=hours, y=solar_gen, mode='lines', name='Solar Gen (kW)', line=dict(color='#FFB800', width=2)))
    bat_fig.add_trace(go.Scatter(x=hours, y=household_demand, mode='lines', name='Household Demand (kW)', line=dict(color='#ef4444', width=2, dash='dash')))
    bat_fig.add_trace(go.Scatter(x=hours, y=battery_charge_state, mode='lines', name='Battery State of Charge (kWh)', line=dict(color='#00E5FF', width=3)))
    bat_fig.update_layout(
        xaxis=dict(title="Hour of Day", tickmode='linear', dtick=2), yaxis=dict(title="Energy (kW / kWh)"), 
        template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(bat_fig, use_container_width=True)

with tab6:
    st.subheader("📄 Automated PDF Commercial Proposal Generator")
    st.write("Generate a downloadable PDF audit report customized with project and client metadata.")

    col_pdf1, col_pdf2 = st.columns(2)
    with col_pdf1:
        client_name = st.text_input("Client/Company Name:", value="Green Energy Corp")
    with col_pdf2:
        kw_rating = st.number_input("Target System Capacity (kWp):", min_value=1.0, value=10.0, step=1.0)

    if st.button("📄 Export Official PDF Proposal"):
        pdf_buffer = generate_pdf_report(
            client_name=client_name,
            kw_rating=kw_rating,
            location=location_name,
            predicted_power=predicted_power,
            daily_kwh=estimated_daily_kwh,
            annual_savings=annual_savings,
            payback_year=payback_year,
            total_25yr_net=total_25yr_net,
            currency=currency
        )
        
        st.success("✅ PDF proposal generated successfully!")
        st.download_button(
            label="📥 Click Here to Download PDF Proposal",
            data=pdf_buffer,
            file_name=f"Solar_Proposal_{client_name.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )