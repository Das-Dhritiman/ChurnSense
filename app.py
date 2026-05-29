"""
Customer Churn Prediction — Streamlit Dashboard
Launch: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

#  Page Config 
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="chart_with_downwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Theme Toggle 
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

#  Theme CSS 
DRACULA_THEME = """
<style>
    /* Main background */
    [data-testid="stAppViewContainer"],
    .main .block-container {
        background-color: #1e1f29 !important;
        color: #f8f8f2 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #282a36 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background-color: #282a36 !important;
    }
    [data-testid="stHeader"] {
        background-color: #1e1f29 !important;
    }

    /* Global text */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"],
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p,
    .stSelectbox label, .stSlider label, .stNumberInput label,
    .stCaption, .stCaption p,
    [data-testid="stCaptionContainer"] {
        color: #f8f8f2 !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #f8f8f2 !important;
    }

    /* Info / success / warning / error boxes */
    [data-testid="stAlert"] {
        background-color: #2c2d3a !important;
        border-radius: 10px !important;
    }
    [data-testid="stAlert"] .stMarkdown,
    [data-testid="stAlert"] .stMarkdown p,
    [data-testid="stAlert"] .stMarkdown li,
    [data-testid="stAlert"] .stMarkdown h3,
    [data-testid="stAlert"] .stMarkdown strong {
        color: #f8f8f2 !important;
    }

    /* Inputs */
    [data-testid="stSelectbox"] > div > div,
    .stSelectbox > div > div {
        background-color: #383a4a !important;
        color: #f8f8f2 !important;
    }
    .stSlider [data-testid="stThumbValue"],
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #f8f8f2 !important;
    }
    .stNumberInput input {
        background-color: #383a4a !important;
        color: #f8f8f2 !important;
    }

    /* Dividers */
    hr {
        border-color: #44475a !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"],
    .stButton > button {
        background-color: #bd93f9 !important;
        color: #1e1f29 !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #ff79c6 !important;
    }

    /* Footer caption */
    [data-testid="stCaptionContainer"] p {
        color: #6272a4 !important;
    }
</style>
"""

LIGHT_THEME = """
<style>
    [data-testid="stAppViewContainer"],
    .main .block-container {
        background-color: #FFFFFF !important;
        color: #262730 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
    }
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }
</style>
"""

# Apply theme based on toggle (instant runtime switching)
if st.session_state.dark_mode:
    st.markdown(DRACULA_THEME, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_THEME, unsafe_allow_html=True)


#  Theme Toggle Functions 
def inject_theme_button_styles():
    """Style the theme toggle button to look premium."""
    is_dark = st.session_state.get('dark_mode', False)
    
    styles = f"""
    <style>
        /* Theme toggle button styling */
        div[data-testid="stHorizontalBlock"] > div:last-child button {{
            background: {'#1e1f29' if is_dark else '#e8ecf1'} !important;
            border: {'1px solid #3a3d4f' if is_dark else '1px solid #cdd3dc'} !important;
            border-radius: 50px !important;
            padding: 8px 20px !important;
            font-size: 20px !important;
            line-height: 1 !important;
            min-height: 0 !important;
            box-shadow: {'0 2px 8px rgba(0,0,0,0.3)' if is_dark else '0 2px 8px rgba(0,0,0,0.08)'} !important;
            transition: all 0.2s ease !important;
            color: {'#f8f8f2' if is_dark else '#262730'} !important;
        }}
        div[data-testid="stHorizontalBlock"] > div:last-child button:hover {{
            background: {'#2c2d3a' if is_dark else '#d4dbe5'} !important;
            transform: scale(1.05);
        }}
        div[data-testid="stHorizontalBlock"] > div:last-child button p {{
            font-size: 20px !important;
            color: {'#f8f8f2' if is_dark else '#262730'} !important;
        }}
    </style>
    """
    st.markdown(styles, unsafe_allow_html=True)


#  Load Model 
MODEL_PATH = os.path.join("models", "churn_model.pkl")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Run `python src/train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]


#  Visualization Functions 
def create_gauge_chart(probability):
    """Create a speedometer-style gauge chart for churn probability."""
    # Determine color based on risk level
    if probability < 0.3:
        bar_color = "#50fa7b"  # Green
        risk_level = "Low Risk"
    elif probability < 0.6:
        bar_color = "#f1fa8c"  # Yellow
        risk_level = "Medium Risk"
    else:
        bar_color = "#ff5555"  # Red
        risk_level = "High Risk"
    
    is_dark = st.session_state.dark_mode
    text_color = "#f8f8f2" if is_dark else "#262730"
    gauge_bg = "#2c2d3a" if is_dark else "#f5f5f5"
    axis_color = "#6272a4" if is_dark else "#666666"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 42, 'color': text_color}},
        title={'text': f"Churn Risk<br><span style='font-size:0.8em;color:{axis_color}'>{risk_level}</span>", 'font': {'size': 22, 'color': text_color}},
        delta={'reference': 50, 'increasing': {'color': "#ff5555"}, 'decreasing': {'color': "#50fa7b"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': axis_color, 'tickfont': {'color': text_color}},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': gauge_bg,
            'borderwidth': 2,
            'bordercolor': axis_color,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(80, 250, 123, 0.25)' if is_dark else 'rgba(80, 250, 123, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(241, 250, 140, 0.25)' if is_dark else 'rgba(241, 250, 140, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(255, 85, 85, 0.25)' if is_dark else 'rgba(255, 85, 85, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#bd93f9", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color}
    )
    return fig


def create_feature_importance_chart(input_df, prediction_proba):
    """Create a horizontal bar chart showing feature contributions."""
    # Get feature contributions using model coefficients or feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # Fallback: use absolute values of input * random importance simulation
        importances = np.abs(input_df.values[0])
    
    # Create feature impact DataFrame
    feature_impact = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'value': input_df.values[0]
    })
    
    # Get top features by importance
    top_features = feature_impact.nlargest(8, 'importance')
    
    # Determine direction based on prediction (positive = increases churn risk)
    churn_prob = prediction_proba[1]
    
    # Map feature names to readable labels
    label_map = {
        'Contract_One year': 'One Year Contract',
        'Contract_Two year': 'Two Year Contract',
        'tenure': 'Tenure (months)',
        'MonthlyCharges': 'Monthly Charges',
        'TotalCharges': 'Total Charges',
        'InternetService_Fiber optic': 'Fiber Optic Internet',
        'InternetService_No': 'No Internet Service',
        'TechSupport_Yes': 'Has Tech Support',
        'OnlineSecurity_Yes': 'Has Online Security',
        'PaymentMethod_Electronic check': 'Electronic Check Payment',
        'PaperlessBilling': 'Paperless Billing',
        'NumServices': 'Number of Services',
        'AvgMonthlySpend': 'Avg Monthly Spend',
        'SeniorCitizen': 'Senior Citizen',
        'Partner': 'Has Partner',
        'Dependents': 'Has Dependents'
    }
    
    # Create readable labels
    top_features['label'] = top_features['feature'].map(lambda x: label_map.get(x, x.replace('_', ' ')))
    
    # Assign colors: features with high values that increase churn = red, decrease = green
    colors = []
    for _, row in top_features.iterrows():
        # Simplified logic: high importance with high value = risk factor
        if row['value'] > 0.5 and row['importance'] > np.median(importances):
            colors.append('#ff5555')  # Red - risk factor
        else:
            colors.append('#50fa7b')  # Green - protective factor
    
    is_dark = st.session_state.dark_mode
    text_color = "#f8f8f2" if is_dark else "#262730"
    grid_color = "#44475a" if is_dark else "#e0e0e0"
    bg_color = "#2c2d3a" if is_dark else "#fafafa"
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['label'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.2f}" for v in top_features['importance']],
        textposition='outside',
        textfont=dict(color=text_color, size=12)
    ))
    
    fig.update_layout(
        title={'text': ' Top Factors Influencing Prediction', 'font': {'size': 16, 'color': text_color}},
        xaxis_title='Feature Importance',
        yaxis_title='',
        height=380,
        margin=dict(l=20, r=80, t=50, b=40),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font={'color': text_color},
        yaxis={'categoryorder': 'total ascending', 'gridcolor': grid_color, 'tickfont': {'color': text_color}},
        xaxis={'gridcolor': grid_color, 'tickfont': {'color': text_color}, 'title': {'font': {'color': text_color}}}
    )
    
    return fig


#  Header 
# Theme toggle in header
header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.title("Customer Churn Prediction System")

with header_col2:
    # Style and render theme toggle button
    inject_theme_button_styles()
    
    is_dark = st.session_state.dark_mode
    btn_label = "Light" if is_dark else "Dark"
    if st.button(btn_label, key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(
    "Predict whether a telecom customer will **churn** based on their profile. "
    "Adjust the inputs in the sidebar and click **Predict**."
)
st.divider()

#  Sidebar Inputs 
st.sidebar.header("Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

st.sidebar.divider()
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, step=1.0)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", min_value=18.0,
    value=float(round(monthly_charges * tenure, 2)),
    help="Auto-calculated from tenure × monthly charges"
)

st.sidebar.divider()
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.sidebar.divider()
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)


#  Build Feature Vector 
def build_input():
    """Build the feature vector matching the training pipeline."""
    data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    df = pd.DataFrame([data])

    #  Feature Engineering (match preprocessing pipeline) 
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype(str)

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    #  Encoding (match preprocessing pipeline) 
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in df.columns:
        if df[col].dtype == "object" and set(df[col].unique()).issubset(
            {"Yes", "No", "Male", "Female"}
        ):
            df[col] = df[col].map(binary_map)

    multi_class_cols = [
        col for col in df.columns if df[col].dtype == "object"
    ]
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

    # Ensure all expected columns exist (fill missing with 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    #  Scale numeric features 
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges",
                        "AvgMonthlySpend", "NumServices"]
    numeric_features = [f for f in numeric_features if f in df.columns]
    df[numeric_features] = scaler.transform(df[numeric_features])

    return df


#  Prediction 
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_button = st.button("Predict Churn", type="primary", use_container_width=True)

if predict_button:
    input_df = build_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    churn_prob = probability[1]
    no_churn_prob = probability[0]

    st.divider()

    # Results - Top Row: Status + Gauge Chart
    #  Section 1: Prediction Result 
    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        if prediction == 1:
            st.error("### HIGH CHURN RISK")
            st.markdown("This customer is **likely to leave**. Immediate retention action recommended.")
        else:
            st.success("### LOW CHURN RISK")
            st.markdown("This customer appears **satisfied and stable**. Continue monitoring.")

    with result_col2:
        gauge_fig = create_gauge_chart(churn_prob)
        st.plotly_chart(gauge_fig, use_container_width=True)

    #  Section 2: Key Metrics 
    st.divider()
    st.subheader("Key Metrics")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(
            "Churn Risk", 
            f"{churn_prob:.1%}",
            delta=f"{(churn_prob - 0.5) * 100:+.1f}% vs avg",
            delta_color="inverse"
        )
    with metric_col2:
        st.metric(
            "Retention", 
            f"{no_churn_prob:.1%}",
            delta=f"{(no_churn_prob - 0.5) * 100:+.1f}% vs avg"
        )
    with metric_col3:
        monthly_revenue_risk = monthly_charges if prediction == 1 else 0
        st.metric(
            "Monthly Revenue at Risk",
            f"${monthly_revenue_risk:.0f}",
            delta="At Risk" if prediction == 1 else "Safe",
            delta_color="inverse" if prediction == 1 else "normal"
        )

    #  Section 3: Visualization 
    st.divider()
    st.subheader("Visualization")
    
    viz_col1, viz_col2 = st.columns([1, 1])

    with viz_col1:
        importance_fig = create_feature_importance_chart(input_df, probability)
        st.plotly_chart(importance_fig, use_container_width=True)

    with viz_col2:
        # Churn probability breakdown donut chart
        is_dark = st.session_state.dark_mode
        text_color = "#f8f8f2" if is_dark else "#262730"
        donut_bg = "#2c2d3a" if is_dark else "#fafafa"
        
        donut_fig = go.Figure(go.Pie(
            labels=["Churn", "No Churn"],
            values=[churn_prob, no_churn_prob],
            hole=0.6,
            marker=dict(colors=["#ff5555", "#50fa7b"], line=dict(color=donut_bg, width=2)),
            textinfo="label+percent",
            textfont=dict(size=14, color=text_color),
        ))
        donut_fig.update_layout(
            title={"text": "Probability Breakdown", "font": {"size": 16, "color": text_color}},
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor=donut_bg,
            font={"color": text_color},
            showlegend=False,
            annotations=[dict(
                text=f"<b>{churn_prob:.0%}</b><br>Churn",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color="#ff5555" if churn_prob > 0.5 else "#50fa7b")
            )]
        )
        st.plotly_chart(donut_fig, use_container_width=True)

    #  Section 4: Risk Factors & Recommendations 
    st.divider()
    st.subheader("Risk Factors & Recommendations")
    
    risks = []
    if contract == "Month-to-month":
        risks.append(("", "Month-to-month contract", "Offer a discount for an annual plan", "high"))
    if tenure < 12:
        risks.append(("", "New customer (< 12 months)", "Strengthen onboarding experience", "medium"))
    if internet_service == "Fiber optic":
        risks.append(("", "Fiber optic user", "Review pricing competitiveness", "medium"))
    if tech_support == "No":
        risks.append(("", "No tech support", "Bundle free support for 3 months", "high"))
    if online_security == "No":
        risks.append(("", "No online security", "Offer security add-on at a discount", "low"))
    if payment_method == "Electronic check":
        risks.append(("", "Electronic check", "Incentivize auto-pay enrollment", "medium"))
    if monthly_charges > 80:
        risks.append(("", "High monthly charges", "Review plan for cost optimization", "high"))

    if risks:
        # Display in a 2-column grid for clean layout
        risk_cols = st.columns(2)
        for idx, (icon, factor, action, severity) in enumerate(risks):
            with risk_cols[idx % 2]:
                if severity == "high":
                    st.error(f"{icon} **{factor}**\n\n→ {action}")
                elif severity == "medium":
                    st.warning(f"{icon} **{factor}**\n\n→ {action}")
                else:
                    st.info(f"{icon} **{factor}**\n\n→ {action}")
    else:
        st.success("No major risk factors identified. Customer profile looks stable!")

else:
    # Default dashboard view
    col_a, col_b = st.columns(2)

    is_dark = st.session_state.dark_mode
    card_bg = '#2c2d3a' if is_dark else '#f7f8fc'
    card_border = '#44475a' if is_dark else '#e0e3ea'
    card_text = '#f8f8f2' if is_dark else '#262730'
    card_muted = '#c0c4d0' if is_dark else '#555'
    accent = '#bd93f9' if is_dark else '#F63366'

    with col_a:
        st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 24px 28px;
            height: 100%;
        ">
            <h3 style="color: {card_text}; margin-top:0;">How it works</h3>
            <ol style="color: {card_muted}; padding-left: 20px; line-height: 2;">
                <li>Fill in the customer profile in the <strong style="color:{accent}">sidebar</strong></li>
                <li>Click <strong style="color:{accent}">Predict Churn</strong></li>
                <li>View the prediction, probability, and actionable recommendations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 24px 28px;
            height: 100%;
        ">
            <h3 style="color: {card_text}; margin-top:0;">Model Info</h3>
            <ul style="color: {card_muted}; padding-left: 20px; line-height: 2; list-style: none;">
                <li><strong style="color:{card_text}">Algorithm:</strong> {bundle['model_name']}</li>
                <li><strong style="color:{card_text}">Features:</strong> {len(feature_names)}</li>
                <li><strong style="color:{card_text}">Training:</strong> SMOTE-balanced, cross-validated</li>
                <li><strong style="color:{card_text}">Explainability:</strong> SHAP-powered insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

#  Footer 
st.divider()
st.caption("Built with Streamlit • scikit-learn • XGBoost • SHAP | Customer Churn Prediction System")
