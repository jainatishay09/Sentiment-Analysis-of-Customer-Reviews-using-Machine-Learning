import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 25px 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #151b2b, #20283d);
    border: 1px solid #30384d;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
    color: white;
}

.hero p {
    font-size: 17px;
    color: #b8c0d0;
}

.section {
    margin-top: 30px;
    margin-bottom: 15px;
}

.section h2 {
    color: white;
}

.metric-card {
    background: #171c27;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #2b3344;
    text-align: center;
}

.metric-title {
    color: #9ca6b8;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 28px;
    font-weight: bold;
}

.info-box {
    background: #151b28;
    padding: 18px;
    border-radius: 15px;
    border-left: 4px solid #4da3ff;
}

.footer {
    text-align: center;
    color: #70798a;
    padding: 30px 0 10px 0;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("Dataset/amazonreviews.csv")


# ---------------------------------------------------------
# SENTIMENT FUNCTION
# ---------------------------------------------------------
def get_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


df["Sentiment"] = df["Rating"].apply(get_sentiment)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">

<h1>📊 Sentiment Analysis of Customer Product Reviews</h1>

<p>
An interactive dashboard for analyzing customer reviews,
ratings, sentiment patterns, locations and machine learning
model performance.
</p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# KPI CALCULATIONS
# ---------------------------------------------------------
total_reviews = len(df)

average_rating = df["Rating"].mean()

positive_reviews = (df["Sentiment"] == "Positive").sum()
negative_reviews = (df["Sentiment"] == "Negative").sum()
neutral_reviews = (df["Sentiment"] == "Neutral").sum()

positive_percent = (positive_reviews / total_reviews) * 100
negative_percent = (negative_reviews / total_reviews) * 100
neutral_percent = (neutral_reviews / total_reviews) * 100


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
st.markdown('<div class="section"><h2>📌 Key Performance Indicators</h2></div>',
            unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Total Reviews</div>
        <div class="metric-value">{total_reviews}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Average Rating</div>
        <div class="metric-value">⭐ {average_rating:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Positive Reviews</div>
        <div class="metric-value">{positive_reviews}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Negative Reviews</div>
        <div class="metric-value">{negative_reviews}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Neutral Reviews</div>
        <div class="metric-value">{neutral_reviews}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
with st.sidebar:

    st.title("🔎 Review Filters")

    st.markdown("---")

    selected_location = st.selectbox(
        "📍 Select Location",
        ["All"] + sorted(df["location"].unique().tolist())
    )

    selected_rating = st.selectbox(
        "⭐ Select Rating",
        ["All"] + sorted(df["Rating"].unique().tolist())
    )

    selected_sentiment = st.selectbox(
        "💬 Select Sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )

    st.markdown("---")

    st.info(
        "Use the filters above to explore reviews "
        "based on location, rating and sentiment."
    )


# ---------------------------------------------------------
# SENTIMENT & RATING DISTRIBUTION
# ---------------------------------------------------------
st.markdown(
    '<div class="section"><h2>📈 Sentiment & Rating Analysis</h2></div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    sentiment_counts = df["Sentiment"].value_counts()

    st.subheader("💬 Sentiment Distribution")

    st.bar_chart(
        sentiment_counts,
        use_container_width=True
    )

with col2:

    rating_counts = df["Rating"].value_counts().sort_index()

    st.subheader("⭐ Rating Distribution")

    st.bar_chart(
        rating_counts,
        use_container_width=True
    )


# ---------------------------------------------------------
# LOCATION ANALYSIS
# ---------------------------------------------------------
st.markdown(
    '<div class="section"><h2>🌍 Location Analysis</h2></div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    location_counts = df["location"].value_counts()

    st.subheader("📍 Reviews by Location")

    st.bar_chart(
        location_counts,
        use_container_width=True
    )

with col2:

    sentiment_location = pd.crosstab(
        df["location"],
        df["Sentiment"]
    )

    st.subheader("💬 Sentiment by Location")

    st.bar_chart(
        sentiment_location,
        use_container_width=True
    )


# ---------------------------------------------------------
# SENTIMENT PERCENTAGE
# ---------------------------------------------------------
st.markdown(
    '<div class="section"><h2>📊 Sentiment Percentage</h2></div>',
    unsafe_allow_html=True
)

percentage_data = pd.DataFrame({
    "Sentiment": [
        "Positive",
        "Negative",
        "Neutral"
    ],
    "Percentage": [
        positive_percent,
        negative_percent,
        neutral_percent
    ]
})

st.bar_chart(
    percentage_data.set_index("Sentiment"),
    use_container_width=True
)


# ---------------------------------------------------------
# MODEL ACCURACY
# ---------------------------------------------------------
st.markdown(
    '<div class="section"><h2>🤖 Machine Learning Model Performance</h2></div>',
    unsafe_allow_html=True
)

model_accuracy = pd.DataFrame({
    "Model": [
        "SVM",
        "Naive Bayes",
        "Decision Tree"
    ],
    "Accuracy": [
        60,
        60,
        45
    ]
})

col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("Model Accuracy Comparison")

    st.bar_chart(
        model_accuracy.set_index("Model"),
        use_container_width=True
    )

with col2:

    best_model = model_accuracy.loc[
        model_accuracy["Accuracy"].idxmax()
    ]

    st.markdown(
        f"""
        <div class="info-box">

        <h3>🏆 Best Performing Model</h3>

        <h2>{best_model["Model"]}</h2>

        <p style="font-size:22px;">
        Accuracy: <b>{best_model["Accuracy"]}%</b>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# REVIEW EXPLORER
# ---------------------------------------------------------
st.markdown(
    '<div class="section"><h2>🔍 Review Explorer</h2></div>',
    unsafe_allow_html=True
)

filtered_df = df.copy()


if selected_location != "All":

    filtered_df = filtered_df[
        filtered_df["location"] == selected_location
    ]


if selected_rating != "All":

    filtered_df = filtered_df[
        filtered_df["Rating"] == selected_rating
    ]


if selected_sentiment != "All":

    filtered_df = filtered_df[
        filtered_df["Sentiment"] == selected_sentiment
    ]


st.write(
    f"### 📋 Filtered Reviews: {len(filtered_df)}"
)


st.dataframe(
    filtered_df[
        [
            "location",
            "Rating",
            "Sentiment",
            "Summary",
            "Review Text"
        ]
    ],
    use_container_width=True,
    height=400
)


# ---------------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------------
st.markdown(
    '<div class="section"><h2>🗃️ Dataset Preview</h2></div>',
    unsafe_allow_html=True
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">

<hr>

<p>
📊 Sentiment Analysis of Customer Product Reviews
</p>

<p>
NLP • Machine Learning • Data Analytics • Streamlit
</p>

</div>
""", unsafe_allow_html=True)