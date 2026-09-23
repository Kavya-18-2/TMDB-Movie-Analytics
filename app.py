import streamlit as st
import pandas as pd
import joblib


# ==========================================
# TMDB MOVIE ANALYTICS - STREAMLIT APP
# ==========================================

st.set_page_config(
    page_title="TMDB Movie Analytics",
    page_icon="🎬",
    layout="wide"
)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/tmdb_movies.csv")

    df["release_date"] = pd.to_datetime(
        df["release_date"],
        errors="coerce"
    )

    df["release_year"] = df["release_date"].dt.year

    return df


# ------------------------------------------
# Load ML Model
# ------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model/movie_model.pkl")


df = load_data()
model = load_model()


# ==========================================
# TITLE
# ==========================================

st.title("🎬 TMDB Movie Analytics")
st.write(
    "Explore movie data, analyze movie trends, "
    "and predict movie ratings using Machine Learning."
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Movie Analytics")

page = st.sidebar.selectbox(
    "Choose an option",
    [
        "Dashboard",
        "Movie Search",
        "Rating Prediction"
    ]
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("📊 Movie Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Movies",
            len(df)
        )

    with col2:
        st.metric(
            "Average Rating",
            round(df["vote_average"].mean(), 2)
        )

    with col3:
        st.metric(
            "Average Popularity",
            round(df["popularity"].mean(), 2)
        )

    with col4:
        st.metric(
            "Total Revenue",
            f"${df['revenue'].sum():,.0f}"
        )

    st.divider()

    st.subheader("📈 Movies Released by Year")

    movies_per_year = (
        df.groupby("release_year")
        .size()
        .reset_index(name="movie_count")
    )

    st.line_chart(
        movies_per_year.set_index("release_year")
    )

    st.subheader("⭐ Rating Distribution")

    rating_data = (
        df["vote_average"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(rating_data)


# ==========================================
# MOVIE SEARCH
# ==========================================

elif page == "Movie Search":

    st.header("🔎 Search for a Movie")

    search_text = st.text_input(
        "Enter movie title"
    )

    if search_text:

        results = df[
            df["title"]
            .fillna("")
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

        if len(results) > 0:

            display_columns = [
                "title",
                "release_date",
                "vote_average",
                "vote_count",
                "popularity",
                "revenue"
            ]

            st.dataframe(
                results[display_columns],
                use_container_width=True
            )

        else:
            st.warning("No movie found.")


# ==========================================
# RATING PREDICTION
# ==========================================

elif page == "Rating Prediction":

    st.header("🤖 Movie Rating Prediction")

    st.write(
        "Enter movie information to predict "
        "the expected movie rating."
    )

    budget = st.number_input(
        "Budget",
        min_value=0.0,
        value=10000000.0
    )

    revenue = st.number_input(
        "Revenue",
        min_value=0.0,
        value=50000000.0
    )

    popularity = st.number_input(
        "Popularity",
        min_value=0.0,
        value=20.0
    )

    runtime = st.number_input(
        "Runtime (minutes)",
        min_value=0.0,
        value=120.0
    )

    vote_count = st.number_input(
        "Vote Count",
        min_value=0.0,
        value=500.0
    )

    release_year = st.number_input(
        "Release Year",
        min_value=1900,
        max_value=2030,
        value=2026
    )

    if st.button("Predict Rating"):

        input_data = pd.DataFrame({
            "budget": [budget],
            "revenue": [revenue],
            "popularity": [popularity],
            "runtime": [runtime],
            "vote_count": [vote_count],
            "release_year": [release_year]
        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"🎯 Predicted Movie Rating: "
            f"{prediction:.2f} / 10"
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "TMDB Movie Analytics | Python | Pandas | "
    "Scikit-learn | Streamlit"
)