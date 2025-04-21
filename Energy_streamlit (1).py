import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import cohere
import requests

# --- Hardcoded API Keys ---
COHERE_API_KEY = "hrp45QCFFHfJsSMFbWYpPCmEIHAn81WO65D3AZyP"
NEWS_API_KEY = "cb4af6030d854b778fbee6ee091129b5"

co = cohere.Client(COHERE_API_KEY)

# --- Custom Page & Sidebar Style ---
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f0f5;
            padding: 10px;
            border-radius: 5px;
            margin-right: 5px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #004d4d;
            color: white;
        }
        .stTabs [role="tablist"] > div > button {
            font-weight: bold;
            background: linear-gradient(90deg, #00796b, #004d4d);
            color: white;
            border-radius: 5px;
            margin: 2px;
        }
        section[data-testid="stSidebar"] {
            background-color: #001f3f !important;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Data (Relative Path) ---
df = pd.read_csv("cleaned_smart_grid_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Sidebar Filters ---
st.sidebar.header("🔧 Filters & Settings")
energy_sources = st.sidebar.multiselect("Select Energy Sources:", ["Solar", "Wind", "Grid"], default=["Grid"])
date_range = st.sidebar.date_input("Select Date Range:", [df["timestamp"].min().date(), df["timestamp"].max().date()])
real_time_mode = st.sidebar.checkbox("Enable Real-Time Updates")
household_usage = st.sidebar.slider("Your Household Energy Consumption (kWh)", min_value=0, max_value=500, value=50)

# --- Filter Data ---
df_filtered = df[
    (df["timestamp"].dt.date.between(date_range[0], date_range[1])) &
    (
        ((df["solar_power_kw"] > 0) & ("Solar" in energy_sources)) |
        ((df["wind_power_kw"] > 0) & ("Wind" in energy_sources)) |
        ((df["grid_supply_kw"] > 0) & ("Grid" in energy_sources))
    )
]

# --- Tabs Layout ---
tabs = st.tabs(["📊 Dashboard", "💡 Recommendations", "🤖 Chatbot", "📊 Insights", "📈 Forecast & Report", "📰 Energy News"])

# --- Tab 1: Dashboard ---
with tabs[0]:
    st.title("⚡Smart Grid Energy Dashboard⚡")
    st.markdown("### Energy Demand Over Time")
    st.line_chart(df_filtered.set_index("timestamp")["power_consumption_kw"])

    col1, col2 = st.columns(2)
    col1.metric("📊 Avg. Load Demand (kW)", round(df_filtered["power_consumption_kw"].mean(), 2))
    col2.metric("💰 Avg. Electricity Price (USD/kWh)", round(df_filtered["electricity_price_usdkwh"].mean(), 4))

    st.write(f"🏠 Your Household Consumption: **{household_usage} kWh**")

    if real_time_mode:
        st.warning("⚡ Real-Time Mode is ON (Will be enabled after EC2 deployment)")
    else:
        st.info("📊 Viewing historical data")

    if st.sidebar.checkbox("Show Raw Data"):
        st.write("### Raw Data")
        st.dataframe(df_filtered, use_container_width=True)

# --- Tab 2: Recommendations ---
with tabs[1]:
    st.header("💡 Personalized Energy Recommendations")
    prompt = f"""You are an expert energy assistant. Analyze this summary of energy consumption data:\n\n{df_filtered.describe().round(2).to_string()}\n\nProvide 3 personalized energy-saving recommendations for a user who consumes {household_usage} kWh."""
    if st.button("Generate Recommendations"):
        with st.spinner("Thinking..."):
            response = co.chat(message=prompt)
            st.success("Here are your recommendations:")
            st.write(response.text.strip())

# --- Tab 3: Chatbot ---
with tabs[2]:
    st.header("🤖 Energy Chatbot")
    st.markdown("Ask the assistant about your energy data. Example prompts:")
    st.markdown("- When was solar generation highest?\n- Which days had high grid dependency?\n- How does wind usage compare to solar?\n- Why is electricity more expensive on some days?")
    user_question = st.text_input("Ask a question about the dataset:", value="")
    if user_question:
        chat_prompt = f"""You are a helpful assistant that analyzes energy consumption data.\n\nData Summary:\n{df_filtered.describe().round(2).to_string()}\n\nUser Question: {user_question}"""
        with st.spinner("Answering..."):
            response = co.chat(message=chat_prompt)
            st.write(response.text.strip())

# --- Tab 4: Visual Insights ---
with tabs[3]:
    st.header("📊 Visual Insights")

    st.subheader("Correlation Heatmap")
    corr = df_filtered.select_dtypes(include=['float64', 'int64']).corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={'shrink': .6})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)

    st.subheader("Solar vs Wind Power")
    fig2, ax2 = plt.subplots()
    df_filtered.set_index("timestamp")[["solar_power_kw", "wind_power_kw"]].plot(ax=ax2)
    ax2.set_ylabel("kW")
    ax2.set_title("Comparison of Solar and Wind Power")
    st.pyplot(fig2)

    st.subheader("Daily Avg. Power Consumption")
    daily_avg = df_filtered.groupby(df_filtered["timestamp"].dt.date)["power_consumption_kw"].mean()
    st.line_chart(daily_avg)

# --- Tab 5: Forecast & Report ---
with tabs[4]:
    st.header("📈 Forecast & PDF Report")
    st.markdown("Generate a simple forecast and download a PDF summary report of this session.")

    future_days = st.slider("Forecast next N days", min_value=1, max_value=30, value=7)
    last_known = df_filtered.groupby(df_filtered["timestamp"].dt.date)["power_consumption_kw"].mean().iloc[-1]
    forecast = [last_known * (1 + 0.01 * i) for i in range(1, future_days + 1)]
    future_dates = pd.date_range(start=df_filtered["timestamp"].max() + pd.Timedelta(days=1), periods=future_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast (kW)": forecast})

    st.subheader("Forecast Plot")
    fig3, ax3 = plt.subplots()
    ax3.plot(forecast_df["Date"], forecast_df["Forecast (kW)"], marker='o')
    ax3.set_title("Forecasted Power Consumption")
    ax3.set_ylabel("kW")
    ax3.set_xlabel("Date")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Smart Grid Forecast Report", ln=True, align="C")
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=forecast_df.to_string(index=False))
        pdf.output("forecast_report.pdf")
        with open("forecast_report.pdf", "rb") as f:
            btn = st.download_button(
                label="Download Report as PDF",
                data=f,
                file_name="forecast_report.pdf",
                mime="application/pdf"
            )

# --- Tab 6: Energy News ---
with tabs[5]:
    st.header("📰 Latest Energy News")
    try:
        news_url = f"https://newsapi.org/v2/everything?q=renewable%20energy&apiKey={NEWS_API_KEY}"
        response = requests.get(news_url)
        if response.status_code == 200:
            news_data = response.json()
            for article in news_data["articles"][:5]:
                st.subheader(article["title"])
                st.write(article["description"])
                st.markdown(f"[Read more]({article['url']})")
        else:
            st.error("Failed to fetch news. Please check the API key or try again later.")
    except:
        st.warning("News section unavailable. External connection issue or API key missing.")

# --- Footer ---
st.caption("Built with ❤️ by Henning Austrup – Energy Enthusiast")
