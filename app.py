import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO

st.set_page_config(page_title="Smart City RAG Assistant", page_icon="🚦", layout="wide")

st.title("🚦 AI Smart City Intelligence Assistant")
st.write("RAG-based assistant for traffic and pollution analysis.")

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

uploaded_file = st.file_uploader("Upload traffic_data.csv or use sample data", type=["csv"])

if uploaded_file is not None:
    if uploaded_file.size == 0:
        st.warning("Uploaded file is empty. Using sample data instead.")
        df = pd.read_csv(StringIO(sample_data))
    else:
        df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using sample smart city data.")
    df = pd.read_csv(StringIO(sample_data))

st.success("Data loaded successfully!")

st.subheader("📊 Dataset")
st.dataframe(df)

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Average Traffic", round(df["Traffic_Count"].mean(), 2))
col3.metric("Average PM2.5", round(df["Pollution_PM2.5"].mean(), 2))

st.subheader("📈 Traffic Count by Location")
st.plotly_chart(px.bar(df, x="Location", y="Traffic_Count", color="Location"), use_container_width=True)

st.subheader("🌫️ Pollution Level by Location")
st.plotly_chart(px.bar(df, x="Location", y="Pollution_PM2.5", color="Location"), use_container_width=True)

st.subheader("🔗 Traffic vs Pollution")
st.plotly_chart(
    px.scatter(df, x="Traffic_Count", y="Pollution_PM2.5", color="Location", size="Traffic_Count"),
    use_container_width=True
)

documents = []
for _, row in df.iterrows():
    documents.append(
        f"At {row['Time']} in {row['Location']}, traffic count was {row['Traffic_Count']} vehicles, "
        f"pollution level was {row['Pollution_PM2.5']} PM2.5, average speed was {row['Avg_Speed']} km/h, "
        f"and weather was {row['Weather']}."
    )

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

st.subheader("🤖 Ask the Smart City Assistant")
query = st.text_input("Ask a question:")

if query:
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    retrieved_data = [documents[i] for i in top_indices]

    st.subheader("AI Answer")
    q = query.lower()

    if "highest traffic" in q or "most congested" in q:
        row = df.loc[df["Traffic_Count"].idxmax()]
        st.write(f"The most congested area is **{row['Location']}** at **{row['Time']}** with **{row['Traffic_Count']} vehicles**.")

    elif "highest pollution" in q or "most polluted" in q:
        row = df.loc[df["Pollution_PM2.5"].idxmax()]
        st.write(f"The highest pollution is at **{row['Location']}** at **{row['Time']}** with PM2.5 level **{row['Pollution_PM2.5']}**.")

    elif "traffic affect pollution" in q or "traffic and pollution" in q or "correlation" in q:
        corr = df["Traffic_Count"].corr(df["Pollution_PM2.5"])
        st.write(f"The correlation between traffic and pollution is **{corr:.2f}**, so higher traffic generally increases pollution.")

    elif "suggest" in q or "reduce" in q or "solution" in q:
        st.write("""
        Suggestions:
        1. Improve traffic signal timing.
        2. Promote public transport.
        3. Encourage carpooling.
        4. Restrict heavy vehicles during peak hours.
        5. Monitor high-pollution zones regularly.
        """)

    else:
        st.write("Relevant retrieved records:")
        for item in retrieved_data:
            st.write(item)

    with st.expander("Retrieved RAG Context"):
        for item in retrieved_data:
            st.write(item)
