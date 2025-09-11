import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools
import warnings
import google.generativeai as genai



def translate_subclusters(subcluster):
    cluster_dict = {
        "Hair Styling & Transformations_0" : "Hair Color & Dyeing",
        "Hair Styling & Transformations_1": "Curly & Wavy Hair Care",
        "Hair Styling & Transformations_3": "Hair Transformations",
        "Hair Styling & Transformations_4": "Hairstyles, Cuts & Accessories",
        "Hair Styling & Transformations_6": "Men's Hair & Grooming",
        "Makeup Tutorials & Challenges_1": "South Asian & Bollywood Makeup",
        "Makeup Tutorials & Challenges_2": "Bridal & Wedding Makeup",
        "Makeup Tutorials & Challenges_3": "Makeup Tutorials, Tips & Techniques",
        "Makeup Tutorials & Challenges_4": "Lipstick & Lip Product Tutorials",
        "Educational Skincare & Wellness_0":"Face Yoga & Anti-Aging Massage",
        "Educational Skincare & Wellness_1":"Skincare Product Reviews & Recommendations",
        "Educational Skincare & Wellness_2":"Men's Skincare & Grooming"
    }

    return cluster_dict[subcluster]


def reverse_translate_subclusters(human_label):
    reverse_dict = {
        "Hair Color & Dyeing": "Hair Styling & Transformations_0",
        "Curly & Wavy Hair Care": "Hair Styling & Transformations_1",
        "Hair Transformations": "Hair Styling & Transformations_3",
        "Hairstyles, Cuts & Accessories": "Hair Styling & Transformations_4",
        "Men's Hair & Grooming": "Hair Styling & Transformations_6",
        "South Asian & Bollywood Makeup": "Makeup Tutorials & Challenges_1", 
        "Bridal & Wedding Makeup":"Makeup Tutorials & Challenges_2",
        "Makeup Tutorials, Tips & Techniques": "Makeup Tutorials & Challenges_3",
        "Lipstick & Lip Product Tutorials": "Makeup Tutorials & Challenges_4",
        "Face Yoga & Anti-Aging Massage": "Educational Skincare & Wellness_0",
        "Skincare Product Reviews & Recommendations": "Educational Skincare & Wellness_1",
        "Men's Skincare & Grooming": "Educational Skincare & Wellness_2"
    }
    
    return reverse_dict[human_label]


def get_subcluster_explanation(human_label: str) -> str:
    """
    Receives a human-readable subcluster label and returns its detailed explanation
    from a hardcoded dictionary.

    Args:
        human_label: The user-friendly name of the subcluster.

    Returns:
        A string containing the detailed explanation for that subcluster,
        or an error message if the label is not found.
    """
    explanation_map = {
        "Hair Color & Dyeing": "The videos in this subcluster are about the process of changing hair color. The content is heavily focused on dyeing, bleaching, highlights, and balayage treatments. The titles frequently mention specific color transformations such as going from black to blonde, dyeing hair pink or purple, and achieving trendy colors like 'espresso brown.' This category includes a mix of professional salon transformations and DIY (do-it-yourself) methods, such as using lemon juice for natural highlights. Overall, this subcluster captures content centered on the chemical and aesthetic alteration of hair color.",
        
        "Curly & Wavy Hair Care": "The videos in this subcluster are about the unique characteristics and styling of curly and wavy hair. A major theme is the 'curly hair journey,' which includes embracing natural texture versus straightening it. The content covers curly hair routines, wash day tutorials, and tips for managing frizz and defining curls. There is also a significant amount of content on creating curls, both with and without heat (e.g., 'heatless curls'), and showcases the versatility and distinct phases of having textured hair.",
        
        "Hair Transformations": "The videos in this subcluster are about showcasing dramatic and often extreme hair makeovers. The core of this category is the 'before and after' reveal, emphasizing a significant change in a person's look. The titles frequently use words like 'transformation,' 'epic,' 'unbelievable,' and 'shocking' to highlight the drastic nature of the haircut or color change. These videos often serve as compilations of satisfying makeovers, hairdresser reactions to transformations, or individual stories of a major style change.",
        
        "Hairstyles, Cuts & Accessories": "The videos in this subcluster are about specific, named hairstyles, popular haircut trends, and the use of hair accessories. Unlike the other categories that focus on a process (like coloring or curling), this one is centered on the final look. Content includes tutorials and showcases of trendy cuts like the 'short Bob,' 'butterfly haircut,' and 'curtain bangs.' It also heavily features content about wigs, hair clips, updos for events like prom, and historical hairstyle trends (e.g., '100 years of hairstyles').",
        
        "Men's Hair & Grooming": "The videos in this subcluster are about men's hairstyling, barbering, and overall grooming. The content is explicitly targeted toward a male audience, featuring popular men's haircuts, hairstyle tutorials according to face shape, and beard styling tips. A recurring theme is 'hair transformation' for men, including the use of hair systems or 'artificial hair' to address hair loss. Beyond just hair, this category also includes broader men's grooming routines for looking more attractive.",

        "South Asian & Bollywood Makeup": "The videos in this subcluster are about makeup looks heavily influenced by South Asian culture, particularly from India. The content frequently involves recreating makeup from Bollywood movies (e.g., Jodha Akbar, Gangubai), tutorials inspired by famous actresses like Priyanka Chopra and Katrina Kaif, and looks for traditional events such as Durga Puja and weddings. This category also includes comparative videos, such as 'India Vs China Makeup Challenge,' and content tailored to a desi audience, reflecting regional beauty trends and celebrity styles.",
        
        "Bridal & Wedding Makeup": "The videos in this subcluster are about makeup for weddings and related events. The content is almost exclusively focused on creating 'bridal looks,' offering step-by-step tutorials for brides, and showcasing dramatic bridal transformations. It also includes makeup tutorials for wedding guests, bridesmaids, and specific ceremonies like the 'haldi' or 'walima.' The themes cover everything from creating a full bridal makeup kit to self-makeup tutorials for one's own wedding events.",
        
        "Makeup Tutorials, Tips & Techniques": "The videos in this subcluster are about a broad range of general makeup tutorials, transformations, and beauty hacks. This category serves as a catch-all for various makeup styles not specific to one theme, including Korean beauty tips, 'no-makeup' makeup looks, anti-aging techniques, and tutorials for different face shapes. The content is educational, focusing on techniques like contouring, blush placement, and creating specific looks like 'dark feminine' or 'natural glam.'",
        
        "Lipstick & Lip Product Tutorials": "The videos in this subcluster are about lip products and lip makeup. The content is highly specific, centering entirely on lipstick, lip liner, and lip gloss. It includes videos on choosing the right lipstick shade for your skin tone or outfit, tutorials for achieving fuller lips, 'lipstick challenges,' and hacks like using one lipstick for a full face of makeup. This category is dedicated to the art and application of lip color.",

        "Face Yoga & Anti-Aging Massage": "The videos in this subcluster are about non-invasive, exercise-based methods for facial rejuvenation and anti-aging. The content is heavily focused on 'face yoga,' 'facial massage,' and 'face fitness' routines designed to naturally lift and tone the skin. Tutorials demonstrate specific exercises to target concerns like wrinkles, smile lines (nasolabial folds), double chins, and sagging jowls. The overall theme is educational and wellness-focused, promoting techniques to achieve a more youthful and sculpted appearance without products or surgery.",
        
        "Skincare Product Reviews & Recommendations": "The videos in this subcluster are about specific skincare products and their benefits. The content centers on product recommendations, reviews, and routines featuring items like serums, moisturizers, sunscreens, and body lotions. Many videos highlight a key ingredient (like Vitamin C) or a specific outcome (like 'glowing skin' or 'skin transformation'). This category includes celebrity-endorsed products and brand features, aiming to educate viewers on which products can help them achieve their desired skin goals.",
        
        "Men's Skincare & Grooming": "The videos in this subcluster are about skincare and grooming specifically for a male audience. The content focuses on 'men's grooming' routines, offering tips on how to achieve clear skin, groom eyebrows, and select the right products. Videos often present 'glow up' guides for men, framing a consistent skincare regimen as a key step towards becoming more attractive. This category addresses the unique skincare needs and interests of men, from basic face washes to more advanced anti-aging solutions."
    }

    # .get() is a safe way to access a dictionary key.
    # It returns the value if the key exists, or the default value otherwise.
    error_message = f"Error: The label '{human_label}' is not a valid subcluster name."
    return explanation_map.get(human_label, error_message)




def card_values(final_signals_df,subcluster_name):
    subcluster = reverse_translate_subclusters(subcluster_name)
    temp_df = final_signals_df[final_signals_df["subcluster"]==subcluster]
    temp_df = temp_df.reset_index()
    recent_6_sum = temp_df.iloc[0]["recent_6_sum"]
    recent_3_sum = temp_df.iloc[0]["recent_3_sum"]
    recent_1_sum = temp_df.iloc[0]["recent_1_sum"]

    print("recent 6 sum : ",recent_6_sum)
    print("recent 3 sum : ",recent_3_sum)
    print("recent 1 sum : ",recent_1_sum)

    return recent_1_sum, recent_3_sum, recent_6_sum

def prob_graph_values(final_signals_df, subcluster_name, month):
    subcluster = reverse_translate_subclusters(subcluster_name)
    temp_df = final_signals_df[final_signals_df["subcluster"] == subcluster].reset_index(drop=True)
    
    values = {
        f"Amount of videos published last {month} month (Dashed grey line)": temp_df.iloc[0][f"recent_{month}_sum"],
        f"Predicted amount of videos for the next {month} month (raw)": temp_df.iloc[0][f"forecast_{month}_sum"] if f"forecast_{month}_sum" in temp_df.columns else temp_df.iloc[0][f"recent_{month}_sum"],  # fallback
        f"80% confidence interval": temp_df.iloc[0][f"range_ci_{month}"],
        "confidence percentage that videos will increase": temp_df.iloc[0][f"confidence_pct_increase_{month}"],
        "confidence percentage that videos will decrease": temp_df.iloc[0][f"confidence_pct_decrease_{month}"],
        "signal_confidence": temp_df.iloc[0][f"signal_confidence_{month}"] if f"signal_confidence_{month}" in temp_df.columns else temp_df.iloc[0][f"pct_change_vs_recent_{month}"]  # fallback
    }
    
    return values




# ---------- HELPER FUNCTIONS (No changes needed here) ----------

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
                 .count().reset_index().rename(columns={"videoId":"video_count"}))
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

    ci_df = pd.DataFrame({
        "forecast": point_forecast.values,
        "lower_80": lower_80,
        "upper_80": upper_80
    }, index=point_forecast.index)

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
        best_params = {
            'trend': row['trend'],
            'seasonal': row['seasonal'],
            'seasonal_periods': row['seasonal_periods']
        }

        print(f"--- Generating final forecast for: {sub} ---")

        # 1. Prepare the full, up-to-date time series data
        monthly_ts = monthly_counts_for_subcluster1(videos, sub)
        if monthly_ts is None or len(monthly_ts) < 24:
            print(f"  Skipping {sub} due to insufficient data.")
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
        metrics = {}
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
            metrics[f'recent_{h}_sum'] = int(recent_sum)
            metrics[f'forecast_{h}_sum'] = int(forecast_sum)
            metrics[f'pct_change_vs_recent_{h}'] = round(pct_change, 2)
            metrics[f'signal_confidence_{h}'] = signal_confidence

            # Add the four new quantitative confidence columns
            metrics[f'confidence_pct_increase_{h}'] = round(confidence_pct_increase, 1)
            metrics[f'confidence_pct_decrease_{h}'] = round(confidence_pct_decrease, 1)

            metrics[f'range_ci_{h}'] = f"{int(lower_bound_sum)} - {int(upper_bound_sum)}"

        # 5. Combine all results into a single record
        final_results.append({
            'subcluster': sub,
            'cv_avg_mape': row['avg_mape'],
            **metrics, # Unpack the 3-month and 6-month metrics here
            'best_trend': best_params['trend'],
            'best_seasonal': best_params['seasonal'],
            'best_periods': best_params['seasonal_periods']
        })

    return pd.DataFrame(final_results)


# --- EXAMPLE USAGE ---

# Assuming 'videos' DataFrame is loaded and you have already run your CV code to get 'forecasting_summary_df'
# forecasting_summary_df = ... (this is the output from your previous script)

# Now, call the new function to generate the final business-ready dashboard

#forecasting_summary_df = pd.read_csv("Holt_Winters_Parameters_anomaly.csv")
#final_signals_df = generate_final_forecasts_and_signals(videos.copy(), forecasting_summary_df)



#'''



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- HELPER FUNCTION 1: Prepare Monthly Data ---
def monthly_counts_for_subcluster(videos, subcluster, start="2020-01"):
    """Prepares the monthly time series data for a given subcluster."""
    df = videos[videos["subcluster"] == subcluster].copy()
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    monthly = (df.groupby(pd.Grouper(key="publishedAt", freq="M"))["videoId"]
                 .count().reset_index().rename(columns={"videoId":"video_count"}))
    if monthly.empty:
        return None
    monthly = monthly.set_index("publishedAt").asfreq("M", fill_value=0)
    return monthly.loc[start:]


# --- HELPER FUNCTION 2: Handle Anomalies ---
def handle_anomalies(series, trend, seasonal, seasonal_periods, sigma=3.0):
    """Detects and corrects anomalies in a time series based on Holt-Winters."""
    # Return original series if it's too short to model
    if len(series) < seasonal_periods * 2:
        return series, pd.Series(dtype=float)

    model = ExponentialSmoothing(
        series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods
    ).fit(optimized=True)
    
    fitted_values = model.fittedvalues
    residuals = series - fitted_values
    residual_std = np.std(residuals)
    
    upper_bound = fitted_values + sigma * residual_std
    lower_bound = fitted_values - sigma * residual_std
    
    # Identify anomalies
    anomalies = series[(series > upper_bound) | (series < lower_bound)]
    
    # Correct anomalies
    corrected_series = series.copy().astype(float)
    corrected_series[series > upper_bound] = upper_bound[series > upper_bound]
    corrected_series[series < lower_bound] = lower_bound[series < lower_bound]
    
    return corrected_series, anomalies


# --- HELPER FUNCTION 3: Train and Forecast with Bootstrap Confidence Intervals ---
def train_and_forecast_with_bootstrap(series, best_params, forecast_horizon=6, n_sims=1000, ci_level=80):


    """Trains the final model and generates forecasts with bootstrap confidence intervals."""
    model = ExponentialSmoothing(
        series,
        trend=best_params['trend'],
        seasonal=best_params['seasonal'],
        seasonal_periods=best_params['seasonal_periods']
    ).fit(optimized=True)

    # Generate the main point forecast
    point_forecast = model.forecast(forecast_horizon)
    
    # Get in-sample residuals for bootstrapping
    residuals = (series - model.fittedvalues).dropna().values

    # Run simulations
    sims = np.zeros((n_sims, forecast_horizon))
    for i in range(n_sims):
        # Sample with replacement from the historical residuals
        sampled_residuals = np.random.choice(residuals, size=forecast_horizon, replace=True)
        sims[i, :] = point_forecast.values + sampled_residuals
        # Ensure forecasts don't go below zero
        sims[i, :] = np.where(sims[i, :] < 0, 0, sims[i, :])

    # Calculate percentiles for the confidence interval
    alpha = (100 - ci_level) / 2
    lower_bound = np.percentile(sims, alpha, axis=0)
    upper_bound = np.percentile(sims, 100 - alpha, axis=0)

    # Combine into a DataFrame
    ci_df = pd.DataFrame({
        "forecast": point_forecast.values,
        f"lower_{ci_level}": lower_bound,
        f"upper_{ci_level}": upper_bound
    }, index=point_forecast.index)

    return ci_df


def plot_forecast_story(videos_df, final_signals_df, subcluster_name, ci_level=80):
    """
    Generates a detailed "Forecast Story" plot for a specific subcluster.

    This plot includes:
    1. Historical actuals (solid line).
    2. The 6-month forecast (dashed line).
    3. A confidence interval for the forecast (shaded area).
    4. Markers for any historical anomalies that were detected and corrected.

    Args:
        videos_df (pd.DataFrame): The raw DataFrame of all videos.
        final_signals_df (pd.DataFrame): The summary DataFrame with best model parameters.
        subcluster_name (str): The name of the subcluster to visualize.
        ci_level (int): The confidence level for the forecast interval (e.g., 80 or 90).

    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """

    videos_df = videos_df[videos_df["publishedAt"] < "2025-07-01"]
    # --- 1. Data Retrieval and Preparation ---
    
    # Get the historical time series data for the subcluster
    monthly_ts = monthly_counts_for_subcluster(videos_df, subcluster_name)
    if monthly_ts is None:
        print(f"Error: No data found for subcluster '{subcluster_name}'.")
        return None

    # Find the best model parameters from the summary dataframe
    try:
        model_params_row = final_signals_df[final_signals_df['subcluster'] == subcluster_name].iloc[0]
        best_params = {
            'trend': model_params_row['best_trend'],
            'seasonal': model_params_row['best_seasonal'],
            'seasonal_periods': int(model_params_row['best_periods'])
        }
    except IndexError:
        print(f"Error: Could not find model parameters for '{subcluster_name}' in the summary dataframe.")
        return None

    # --- 2. Anomaly Detection ---
    
    # Find and correct anomalies on the full historical dataset
    cleaned_series, detected_anomalies = handle_anomalies(
        monthly_ts["video_count"],
        trend=best_params['trend'],
        seasonal=best_params['seasonal'],
        seasonal_periods=best_params['seasonal_periods']
    )

    # --- 3. Final Forecasting ---
    
    # Train the final model on the CLEANED data and generate the forecast with confidence intervals
    forecast_df = train_and_forecast_with_bootstrap(
        cleaned_series, best_params, forecast_horizon=6, ci_level=ci_level
    )

    # --- 4. Visualization with Plotly ---
    fig = go.Figure()

    # Layer 1: Confidence Interval (shaded area)
    fig.add_trace(go.Scatter(
        # VVVVVV THIS IS THE CORRECTED LINE VVVVVV
        x=forecast_df.index.tolist() + forecast_df.index[::-1].tolist(),
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        y=pd.concat([forecast_df[f'upper_{ci_level}'], forecast_df[f'lower_{ci_level}'][::-1]]),
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name=f'{ci_level}% Confidence Interval'
    ))

    # Layer 2: Historical Data (solid line)
    fig.add_trace(go.Scatter(
        x=monthly_ts.index, 
        y=monthly_ts['video_count'],
        mode='lines+markers', 
        name='Historical Video Count',
        line=dict(color='blue')
    ))

    # Layer 3: Forecast (dashed line)
    fig.add_trace(go.Scatter(
        x=forecast_df.index, 
        y=forecast_df['forecast'],
        mode='lines', 
        name='Forecast', 
        line=dict(color='orange', dash='dash')
    ))

    # Layer 4: Detected Anomalies (red 'X' markers)
    
    # --- 5. Formatting and Layout ---
    
    # Add a vertical line to separate history from forecast
    fig.add_vline(x=monthly_ts.index[-1], line_width=1, line_dash="dash", line_color="grey")

    

    fig.update_layout(
        title=f"<b>Raw 6-Month Forecast & Confidence</b>",
        xaxis_title="Date",
        yaxis_title="Monthly Video Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    return fig


##################################################################################################



from scipy.stats import gaussian_kde

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import gaussian_kde

# --- HELPER FUNCTIONS (Required for the plotting function to work) ---

def monthly_counts_for_subcluster3(videos, subcluster, start="2020-01"):
    df = videos[videos["subcluster"] == subcluster].copy()
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    monthly = (df.groupby(pd.Grouper(key="publishedAt", freq="M"))["videoId"]
                 .count().reset_index().rename(columns={"videoId":"video_count"}))
    if monthly.empty: return None
    monthly = monthly.set_index("publishedAt").asfreq("M", fill_value=0)
    return monthly.loc[start:]

def handle_anomalies3(series, trend, seasonal, seasonal_periods, sigma=3.0):
    if len(series) < seasonal_periods * 2: return series
    model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit(optimized=True)
    fitted_values = model.fittedvalues
    residuals = series - fitted_values
    residual_std = np.std(residuals)
    upper_bound = fitted_values + sigma * residual_std
    lower_bound = fitted_values - sigma * residual_std
    corrected_series = series.copy().astype(float)
    corrected_series[series > upper_bound] = upper_bound[series > upper_bound]
    corrected_series[series < lower_bound] = lower_bound[series < lower_bound]
    return corrected_series

def train_and_get_simulations(series, best_params, forecast_horizon=6, n_sims=1000):
    """
    Trains model and returns the raw bootstrap simulations.
    """
    model = ExponentialSmoothing(
        series, trend=best_params['trend'], seasonal=best_params['seasonal'],
        seasonal_periods=best_params['seasonal_periods']).fit(optimized=True)
    
    point_forecast = model.forecast(forecast_horizon)
    residuals = (series - model.fittedvalues).dropna().values
    if len(residuals) == 0: residuals = np.array([0])

    sims = np.zeros((n_sims, forecast_horizon))
    for i in range(n_sims):
        sampled_residuals = np.random.choice(residuals, size=forecast_horizon, replace=True)
        sims[i, :] = point_forecast.values + sampled_residuals
        sims[i, :] = np.where(sims[i, :] < 0, 0, sims[i, :])
        
    return sims

# --- NEW VISUALIZATION FUNCTION ---

def visualize_distribution_and_test(videos_df, final_df, subcluster_name, horizon):
    """
    Generates a distribution plot (KDE "bell curve") of bootstrap simulations,
    showing the confidence interval, critical regions, and the location of the recent sum.

    Args:
        videos_df (pd.DataFrame): The raw videos data.
        final_df (pd.DataFrame): The final_signals_df containing all forecast results.
        subcluster_name (str): The name of the subcluster to visualize.
        horizon (int): The forecast horizon to plot (1, 3, or 6).
    """
    # --- 1. Data Extraction ---
    try:
        trend_data = final_df.loc[final_df['subcluster'] == subcluster_name].iloc[0]
        params_row = final_df[final_df['subcluster'] == subcluster_name].iloc[0]
    except IndexError:
        print(f"Error: Subcluster '{subcluster_name}' not found.")
        return

    recent_sum = trend_data[f'recent_{horizon}_sum']
    best_params = {
        'trend': params_row['best_trend'], 'seasonal': params_row['best_seasonal'],
        'seasonal_periods': int(params_row['best_periods'])
    }

    # --- 2. Re-run Simulation to Get Raw Data ---
    monthly_ts = monthly_counts_for_subcluster3(videos_df, subcluster_name)
    cleaned_series = handle_anomalies3(monthly_ts["video_count"], **best_params)
    bootstrap_sims = train_and_get_simulations(cleaned_series, best_params)
    simulated_sums = bootstrap_sims[:, :horizon].sum(axis=1) if horizon > 1 else bootstrap_sims[:, 0]

    # --- 3. Prepare Data for Plotting ---
    lower_bound = np.percentile(simulated_sums, 10)
    upper_bound = np.percentile(simulated_sums, 90)
    
    # Create the Kernel Density Estimate (the "bell curve")
    kde = gaussian_kde(simulated_sums)
    x_range = np.linspace(min(simulated_sums.min(), recent_sum) * 0.9, 
                          max(simulated_sums.max(), recent_sum) * 1.1, 500)
    y_kde = kde(x_range)

    fig = go.Figure()

    # --- 4. Plot the Shaded Regions ---
    # Central 80% "Uncertain" Region
    fig.add_trace(go.Scatter(
        x=x_range[(x_range >= lower_bound) & (x_range <= upper_bound)],
        y=y_kde[(x_range >= lower_bound) & (x_range <= upper_bound)],
        fill='tozeroy', mode='lines', line_color='gray', fillcolor='rgba(128,128,128,0.3)',
        name='80% Confidence Interval (Uncertain)'
    ))
    # Lower 10% "High Decrease" Region
    fig.add_trace(go.Scatter(
        x=x_range[x_range < lower_bound], y=y_kde[x_range < lower_bound],
        fill='tozeroy', mode='lines', line_color='red', fillcolor='rgba(255,0,0,0.3)',
        name='Critical Region (Decrease)'
    ))
    # Upper 10% "High Increase" Region
    fig.add_trace(go.Scatter(
        x=x_range[x_range > upper_bound], y=y_kde[x_range > upper_bound],
        fill='tozeroy', mode='lines', line_color='green', fillcolor='rgba(0,255,0,0.3)',
        name='Critical Region (Increase)'
    ))

    # --- 5. Add the Vertical Line for the Recent Sum ---
    fig.add_vline(x=recent_sum, line_width=3, line_dash="dash", line_color="#708090",
                  annotation_text="Recent Sum", annotation_position="top right")

    # --- 6. Formatting ---
    fig.update_layout(
        title=f"<b>{horizon}-Month Forecast Distribution</b>",
        xaxis_title=f"Total Video Count over {horizon} Months",
        yaxis_title="Probability Density",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,  # 👈 make the graph shorter (try 250–350)
        margin=dict(t=60, b=40, l=40, r=20)  # 👈 tighter margins
    )
    
    return fig

# --- Example Usage for the New Visualization ---

# Assume 'videos' and 'final_signals_df' are loaded and prepared


#'''


def explain_chart(prompt: str):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-pro")  # or gemini-1.5-pro
    response = model.generate_content(prompt)
    return response.text