import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO

st.set_page_config(
    page_title="AI Data Insight Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Insight Assistant")
st.write("A RAG-based intelligent assistant for analyzing CSV datasets.")

sample_data = """Time,Location,Traffic_Count,Pollution_PM2.5,Weather,Avg_Speed
08:00,City Center,180,210,Sunny,22
09:00,City Center,260,280,Sunny,14
10:00,Market,220,250,Sunny,16
11:00,Highway,300,230,Cloudy,35
12:00,Industrial Area,350,310,Cloudy,18
13:00,Bus Stand,280,290,Sunny,12
14:00,School Zone,160,170,Sunny,20
15:00,Residential Area,90,110,Cloudy,32
16:00,Market,270,300,Sunny,13
17:00,City Center,390,350,Sunny,9
18:00,Bus Stand,330,340,Cloudy,10
19:00,Industrial Area,370,360,Cloudy,15
"""

uploaded_file = st.file_uploader("Upload any CSV dataset", type=["csv"])

if uploaded_file is not None and uploaded_file.size > 0:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No valid file uploaded. Using sample smart city dataset.")
    df = pd.read_csv(StringIO(sample_data))

st.success("Dataset loaded successfully!")

st.subheader("📊 Dataset Preview")
st.dataframe(df)

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Numeric Columns", len(numeric_cols))

st.subheader("📌 Dataset Summary")
st.write(df.describe())

if numeric_cols:
    st.subheader("📈 Numeric Data Visualization")
    selected_num = st.selectbox("Select numeric column", numeric_cols)

    if categorical_cols:
        selected_cat = st.selectbox("Select category column", categorical_cols)
        fig = px.bar(df, x=selected_cat, y=selected_num, color=selected_cat)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(df, y=selected_num)
        st.plotly_chart(fig, use_container_width=True)

if len(numeric_cols) >= 2:
    st.subheader("🔗 Relationship Analysis")
    x_col = st.selectbox("Select X column", numeric_cols, key="x")
    y_col = st.selectbox("Select Y column", numeric_cols, key="y")

    fig = px.scatter(df, x=x_col, y=y_col)
    st.plotly_chart(fig, use_container_width=True)

    corr = df[x_col].corr(df[y_col])
    st.write(f"Correlation between **{x_col}** and **{y_col}** is **{corr:.2f}**.")

documents = []

 for_index = df.reset_index()

for _, row in for_index.iterrows():
    text = "Record details: "
    for col in df.columns:
        text += f"{col} is {row[col]}, "
    documents.append(text)

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

st.subheader("🤖 Ask Questions About Your Data")
query = st.text_input("Example: Which value is highest? Give summary. Find relationship.")

if query:
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    retrieved = [documents[i] for i in top_indices]

    q = query.lower()

    st.subheader("AI Answer")

    if "summary" in q or "summarize" in q:
        st.write("This dataset contains the following columns:")
        st.write(list(df.columns))

        st.write("Numeric column averages:")
        for col in numeric_cols:
            st.write(f"- **{col}** average: {df[col].mean():.2f}")

    elif "highest" in q or "maximum" in q or "max" in q:
        if numeric_cols:
            chosen_col = st.selectbox("Select column to find highest value", numeric_cols, key="highest")
            row = df.loc[df[chosen_col].idxmax()]
            st.write(f"The highest value of **{chosen_col}** is **{row[chosen_col]}**.")
            st.write("Complete record:")
            st.write(row)
        else:
            st.write("No numeric column found for highest value analysis.")

    elif "lowest" in q or "minimum" in q or "min" in q:
        if numeric_cols:
            chosen_col = st.selectbox("Select column to find lowest value", numeric_cols, key="lowest")
            row = df.loc[df[chosen_col].idxmin()]
            st.write(f"The lowest value of **{chosen_col}** is **{row[chosen_col]}**.")
            st.write("Complete record:")
            st.write(row)
        else:
            st.write("No numeric column found for lowest value analysis.")

    elif "relationship" in q or "correlation" in q or "affect" in q:
        if len(numeric_cols) >= 2:
            st.write("Correlation analysis between numeric columns:")
            corr_matrix = df[numeric_cols].corr()
            st.dataframe(corr_matrix)
        else:
            st.write("At least two numeric columns are needed for relationship analysis.")

    elif "suggest" in q or "recommend" in q or "solution" in q:
        st.write("""
        Recommendations based on data insights:

        1. Identify high-value or high-risk records.
        2. Compare numeric trends using correlation.
        3. Monitor extreme values such as maximum and minimum cases.
        4. Use visualization to support decision-making.
        5. Extend this system with real-time data sources.
        """)

    else:
        st.write("Relevant retrieved records from the dataset:")
        for item in retrieved:
            st.write(item)

    with st.expander("Retrieved RAG Context"):
        for item in retrieved:
            st.write(item)
