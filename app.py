import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools
import warnings
import functions as fc

# Page title (centered, smaller font)
st.markdown(
    """
    <h1 style="text-align:center; font-size:28px;">
        ✨ L'Oréal YouTube Videos Analysis ✨
    </h1>
    """,
    unsafe_allow_html=True
)

# --- Load Data ---
videos = pd.read_csv("videos_subclusters_corrected.csv")
videos.drop(columns=["Unnamed: 0","Unnamed: 0.1"], inplace=True)

good_subclusters = [
    "Hair Styling & Transformations_0","Hair Styling & Transformations_1","Hair Styling & Transformations_3",
    "Hair Styling & Transformations_4","Hair Styling & Transformations_6",
    "Makeup Tutorials & Challenges_1","Makeup Tutorials & Challenges_2","Makeup Tutorials & Challenges_3",
    "Makeup Tutorials & Challenges_4",
    "Educational Skincare & Wellness_0","Educational Skincare & Wellness_1","Educational Skincare & Wellness_2"
]

videos["publishedAt"] = pd.to_datetime(videos["publishedAt"])
videos = videos[videos["publishedAt"] < "2025-07-01"]
videos = videos[videos["subcluster"].isin(good_subclusters)].reset_index(drop=True)


forecasting_summary_df = pd.read_csv("Holt_Winters_Parameters_anomaly.csv")

#st.write("Videos Dataframe")
#st.dataframe(videos)

#st.write("Final_signals_df")
#st.dataframe(forecasting_summary_df)


if "final_signals_df" not in st.session_state:
    st.session_state.final_signals_df = fc.generate_final_forecasts_and_signals(
        videos.copy(), forecasting_summary_df.copy()
    )

final_signals_df = st.session_state.final_signals_df

config = {
    "scrollZoom": False,
    "doubleClick": False,
    "displayModeBar": False
}

st.markdown("---")
# --- Sidebar ---
st.sidebar.header("🔎 Filters")

# Video type dropdown
options = [
    "Hair Styling & Transformations", 
    "Makeup Tutorials & Challenges", 
    "Educational Skincare & Wellness"
]
choice = st.sidebar.selectbox("🎥 Select Video Type:", options)

# Subcluster dropdown (depends on video type)
if choice == "Hair Styling & Transformations":
    options2 = [
        "Hair Color & Dyeing",
        "Curly & Wavy Hair Care",
        "Hair Transformations",
        "Hairstyles, Cuts & Accessories",
        "Men's Hair & Grooming"
    ]
elif choice == "Makeup Tutorials & Challenges":
    options2 = [
        "South Asian & Bollywood Makeup",
        "Bridal & Wedding Makeup",
        "Makeup Tutorials, Tips & Techniques",
        "Lipstick & Lip Product Tutorials"
    ]
elif choice == "Educational Skincare & Wellness":
    options2 = [
        "Face Yoga & Anti-Aging Massage",
        "Skincare Product Reviews & Recommendations",
        "Men's Skincare & Grooming"
    ]
else:
    options2 = []

subchoice = st.sidebar.selectbox("📌 Select Subtype:", options2)

# Your Selection summary box (inside sidebar)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="padding:15px; border-radius:12px; 
                box-shadow:0 2px 6px rgba(0,0,0,0.1);">
        <h4 style="margin-top:0;">📊 Your Selection</h4>
        <p><b>Video Type:</b> {choice}</p>
        <p><b>Video Subtype:</b> {subchoice}</p>
        <p><b>Explanation:</b><br>{fc.get_subcluster_explanation(subchoice)}</p>
    </div>
    """,
    unsafe_allow_html=True
)




s1, s3, s6 = fc.card_values(final_signals_df.copy(), subchoice)

col1, col2, col3 = st.columns(3)

def kpi_card(title, value, emoji):
    st.markdown(
        f"""
        <div style="
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #dcdcdc;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
        ">
            <h5 style="margin-bottom:6px; font-size:14px;">{emoji} {title}</h5>
            <p style="font-size:26px; font-weight:700;margin:0;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col1:
    kpi_card("Videos Published this Month", s1, "📅")

with col2:
    kpi_card("Total Videos (Last 3 Months)", s3, "📊")

with col3:
    kpi_card("Total Videos (Last 6 Months)", s6, "📈")



# Generate the forecast story figure
forecast_story_fig = fc.plot_forecast_story(
    videos_df=videos.copy(),
    final_signals_df=final_signals_df.copy(),
    subcluster_name=fc.reverse_translate_subclusters(subchoice),
    ci_level=80
)

# Display in Streamlit
if forecast_story_fig is not None:
    st.plotly_chart(forecast_story_fig, use_container_width=True, config=config)
else:
    st.warning("⚠️ No forecast available for this subcluster.")


#st.write("Videos Dataframe")
#st.dataframe(videos)

#st.write("Final_signals_df")
#st.dataframe(final_signals_df)

# Detect theme background
from streamlit_theme import st_theme
theme = st_theme()

theme_bg = theme["base"] # "light" or "dark"
print(theme_bg)
# Set color depending on mode
if theme_bg == "dark":
    default_color = "#E0E0E0"  # Light gray text for dark background
else:
    default_color = "#2C3E50"  # Dark navy text for light background

st.subheader("Residual Bootstrapping Simulation Results")


prob_chart_1m =fc.visualize_distribution_and_test(videos.copy(), final_signals_df.copy(), fc.reverse_translate_subclusters(subchoice), horizon=1)

col1, col2 = st.columns([6,3])

with col1:
    st.plotly_chart(prob_chart_1m,use_container_width=True,config=config)

with col2:
    value_dict1 = fc.prob_graph_values(final_signals_df, subchoice, month=1)

    # Build card with all values
    # Start the HTML for the card
    html_string = """
    <div style="
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;">
    """

    # Add each key-value pair to the HTML string
    for k, v in value_dict1.items():
        color = default_color  # Default color

        if k == "signal_confidence":
            if v == "high_increase":
                color = "green"
            elif v == "high_decrease":
                color = "red"
            elif v == "uncertain":
                color = "orange"  # Using orange for better readability than yellow

        html_string += f'<p style="margin: 0;"><b style="font-size:13px;">{k.replace("_", " ").title()}:</b> <span style="font-size:13px; color:{color};">{v}</span></p>'

    # Close the div tag
    html_string += "</div>"

    # Display the card
    st.markdown(html_string, unsafe_allow_html=True)

prob_chart_3m =fc.visualize_distribution_and_test(videos.copy(), final_signals_df.copy(), fc.reverse_translate_subclusters(subchoice), horizon=3)

col1, col2 = st.columns([3,6])

with col1:
    value_dict2 = fc.prob_graph_values(final_signals_df, subchoice, month=3)

    # Build card with all values
    # Start the HTML for the card
    html_string = """
    <div style="
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;">
    """

    # Add each key-value pair to the HTML string
    for k, v in value_dict2.items():
        color = default_color # Default color

        if k == "signal_confidence":
            if v == "high_increase":
                color = "green"
            elif v == "high_decrease":
                color = "red"
            elif v == "uncertain":
                color = "orange"  # Using orange for better readability than yellow

        html_string += f'<p style="margin: 0;"><b style="font-size:13px;">{k.replace("_", " ").title()}:</b> <span style="font-size:13px; color:{color};">{v}</span></p>'

    # Close the div tag
    html_string += "</div>"

    # Display the card
    st.markdown(html_string, unsafe_allow_html=True)
with col2:
    st.plotly_chart(prob_chart_3m,use_container_width=True,config=config)
    

col1, col2 = st.columns([6,3])

prob_chart_6m =fc.visualize_distribution_and_test(videos.copy(), final_signals_df.copy(), fc.reverse_translate_subclusters(subchoice), horizon=6)

with col1:
    st.plotly_chart(prob_chart_6m,use_container_width=True,config=config)

with col2:
    value_dict3 = fc.prob_graph_values(final_signals_df, subchoice, month=6)

    # Build card with all values
    # Start the HTML for the card
    html_string = """
    <div style="
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;">
    """

    # Add each key-value pair to the HTML string
    for k, v in value_dict3.items():
        color = default_color  # Default color

        if k == "signal_confidence":
            if v == "high_increase":
                color = "green"
            elif v == "high_decrease":
                color = "red"
            elif v == "uncertain":
                color = "orange"  # Using orange for better readability than yellow

        html_string += f'<p style="margin: 0;"><b style="font-size:13px;">{k.replace("_", " ").title()}:</b> <span style="font-size:13px; color:{color};">{v}</span></p>'

    # Close the div tag
    html_string += "</div>"

    # Display the card
    st.markdown(html_string, unsafe_allow_html=True)


prompt = f'''
Persona : you ara an AI assistant that will provide explanation to users about forecasted values of the amount of videos of a certain type that will be publish for the next 1,3,6 month, this is the python code

python code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools

import warnings
warnings.filterwarnings("ignore")

# ---------- HELPER FUNCTIONS (No changes needed here) ----------

def calculate_mape(y_true, y_pred):
    """Robust MAPE calculation that ignores zero actuals."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0 if np.all(y_pred == 0) else np.inf
    percentage_error = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    return np.mean(percentage_error) * 100

def monthly_counts_for_subcluster1(videos, subcluster, start="2020-01"):
    """Prepares the monthly time series data for a given subcluster."""
    df = videos[videos["subcluster"] == subcluster].copy()
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    monthly = (df.groupby(pd.Grouper(key="publishedAt", freq="M"))["videoId"]
                 .count().reset_index().rename(columns={{"videoId":"video_count"}}))
    if monthly.empty:
        return None
    monthly = monthly.set_index("publishedAt").asfreq("M", fill_value=0)
    return monthly.loc[start:]

def handle_anomalies1(series, trend, seasonal, seasonal_periods, sigma=3.0):
    """Detects and corrects anomalies in a time series based on Holt-Winters."""
    model = ExponentialSmoothing(
        series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods
    ).fit(optimized=True)
    fitted_values = model.fittedvalues
    residuals = series - fitted_values
    residual_std = np.std(residuals)
    upper_bound = fitted_values + sigma * residual_std
    lower_bound = fitted_values - sigma * residual_std
    anomalies_upper = series > upper_bound
    anomalies_lower = series < lower_bound
    corrected_series = series.copy().astype(float)
    corrected_series[anomalies_upper] = upper_bound[anomalies_upper]
    corrected_series[anomalies_lower] = lower_bound[anomalies_lower]
    return corrected_series

def train_and_forecast_with_bootstrap1(series, best_params, forecast_horizon=6, n_sims=1000):
    """Trains the final model and generates forecasts with bootstrap confidence intervals."""
    model = ExponentialSmoothing(
        series,
        trend=best_params['trend'],
        seasonal=best_params['seasonal'],
        seasonal_periods=best_params['seasonal_periods']
    ).fit(optimized=True)

    point_forecast = model.forecast(forecast_horizon)
    residuals = (series - model.fittedvalues).dropna().values

    sims = np.zeros((n_sims, forecast_horizon))
    for i in range(n_sims):
        sampled_residuals = np.random.choice(residuals, size=forecast_horizon, replace=True)
        sims[i, :] = point_forecast.values + sampled_residuals
        sims[i, :] = np.where(sims[i, :] < 0, 0, sims[i, :])

    # Using 80% confidence for signals (alpha=0.1)
    lower_80 = np.percentile(sims, 10, axis=0)
    upper_80 = np.percentile(sims, 90, axis=0)

    ci_df = pd.DataFrame({{
        "forecast": point_forecast.values,
        "lower_80": lower_80,
        "upper_80": upper_80
    }}, index=point_forecast.index)

    return ci_df, sims

# ---------- REWRITTEN CORE FUNCTION ----------

def generate_final_forecasts_and_signals(videos, forecasting_summary_df, anomaly_sigma=3.0):
    """
    Takes the best parameters for each subcluster, retrains the model on all data (with anomaly handling),
    and generates a final dashboard with 3-month and 6-month trend signals AND a quantitative confidence score.
    """
    #videos = videos[videos["publishedAt"] < "2025-07-01"]
    final_results = []

    # Iterate through the best model found for each subcluster
    for _, row in forecasting_summary_df.iterrows():
        sub = row['subcluster']
        best_params = {{
            'trend': row['trend'],
            'seasonal': row['seasonal'],
            'seasonal_periods': row['seasonal_periods']
        }}

        print(f"--- Generating final forecast for: {{sub}} ---")

        # 1. Prepare the full, up-to-date time series data
        monthly_ts = monthly_counts_for_subcluster1(videos, sub)
        if monthly_ts is None or len(monthly_ts) < 24:
            print(f"  Skipping {{sub}} due to insufficient data.")
            continue

        # 2. Handle anomalies on the FULL dataset before final training
        cleaned_series = handle_anomalies1(
            monthly_ts["video_count"],
            trend=best_params['trend'],
            seasonal=best_params['seasonal'],
            seasonal_periods=best_params['seasonal_periods'],
            sigma=anomaly_sigma
        )

        # 3. Train the final model and generate bootstrap forecasts for 6 months
        forecast_df, bootstrap_sims = train_and_forecast_with_bootstrap1(
            cleaned_series, best_params, forecast_horizon=6, n_sims=1000
        )

        # 4. Calculate metrics for both 6-month and 3-month horizons
        horizons = [6, 3, 1]
        metrics = {{}}
        for h in horizons:
            # Recent history sums
            recent_sum = monthly_ts["video_count"].iloc[-h:].sum()

            # Forecast sums from the point forecast
            forecast_sum = forecast_df["forecast"].iloc[:h].sum()

            # Calculate percentage change
            pct_change = ((forecast_sum - recent_sum) / recent_sum) * 100 if recent_sum > 0 else float('inf')

            # Use bootstrap simulations to get confidence interval of the SUM
            simulated_sums = bootstrap_sims[:, :h].sum(axis=1)
            lower_bound_sum = np.percentile(simulated_sums, 10)
            upper_bound_sum = np.percentile(simulated_sums, 90)

            # Calculate the confidence probability from the simulations
            n_sims = len(simulated_sums)
            increase_count = (simulated_sums > recent_sum).sum()
            decrease_count = (simulated_sums < recent_sum).sum()

            confidence_pct_increase = (increase_count / n_sims) * 100
            confidence_pct_decrease = (decrease_count / n_sims) * 100

            # Determine signal confidence (qualitative label)
            if lower_bound_sum > recent_sum:
                signal_confidence = "high_increase"
            elif upper_bound_sum < recent_sum:
                signal_confidence = "high_decrease"
            else:
                signal_confidence = "uncertain"

            # Store all metrics for this horizon
            metrics[f'recent_{{h}}_sum'] = int(recent_sum)
            metrics[f'forecast_{{h}}_sum'] = int(forecast_sum)
            metrics[f'pct_change_vs_recent_{{h}}'] = round(pct_change, 2)
            metrics[f'signal_confidence_{{h}}'] = signal_confidence

            # Add the four new quantitative confidence columns
            metrics[f'confidence_pct_increase_{{h}}'] = round(confidence_pct_increase, 1)
            metrics[f'confidence_pct_decrease_{{h}}'] = round(confidence_pct_decrease, 1)

            metrics[f'range_ci_{{h}}'] = f"{{int(lower_bound_sum)}} - {{int(upper_bound_sum)}}"

        # 5. Combine all results into a single record
        final_results.append({{
            'subcluster': sub,
            'cv_avg_mape': row['avg_mape'],
            **metrics, # Unpack the 3-month and 6-month metrics here
            'best_trend': best_params['trend'],
            'best_seasonal': best_params['seasonal'],
            'best_periods': best_params['seasonal_periods']
        }})

    return pd.DataFrame(final_results)


# --- EXAMPLE USAGE ---

# Assuming 'videos' DataFrame is loaded and you have already run your CV code to get 'forecasting_summary_df'
# forecasting_summary_df = ... (this is the output from your previous script)

# Now, call the new function to generate the final business-ready dashboard

forecasting_summary_df = pd.read_csv("Holt_Winters_Parameters_anomaly.csv")
final_signals_df = generate_final_forecasts_and_signals(videos.copy(), forecasting_summary_df)


explanation of python code :
in each row in the videos dataframe is a video and its details and the forecasting summary dataframe contains the suitable parameters for a holt winters model

right now the user is analysing the videos in subcluster {subchoice}

and here are the forecasting details of that subcluster

for 1 month forecast here is the forecasting details:
{value_dict1}

for 3 month forecast here is the forecasting details:
{value_dict2}

for 6 month forecast here is the forecasting details:
{value_dict3}

Now  provide an explanation to the user so that they understand the explanation must be easy to understand , your explanation for each of the 1,3,6 month forecast shoudl start with like
"For the X month forecast, from the 1000 residual bootstrap simulations ..."
in your explanation, emphasize more on the confidence interval rather than raw forecast values cuz raw forecast values are not too trutstable

your explanation should include 

Last X Months' Performance, What the Simulations Tell Us, and Why is the signal like that

'''

if st.button("🤔 Don’t understand the graph? Click here for AI explanation"):
    with st.spinner("AI is analyzing the chart..."):
        explanation = fc.explain_chart(prompt)
    st.success("Here’s the explanation:")
    st.write(explanation)






