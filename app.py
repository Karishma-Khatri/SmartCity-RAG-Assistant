import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Smart City RAG Assistant",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Smart City Intelligence Assistant")
st.write("RAG-based assistant for traffic and pollution analysis.")

uploaded_file = st.file_uploader("Upload traffic_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Data uploaded successfully!")

    st.subheader("📊 Dataset")
    st.dataframe(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Average Traffic", round(df["Traffic_Count"].mean(), 2))
    col3.metric("Average PM2.5", round(df["Pollution_PM2.5"].mean(), 2))

    st.subheader("📈 Traffic Count by Location")
    fig1 = px.bar(df, x="Location", y="Traffic_Count", color="Location")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🌫️ Pollution Level by Location")
    fig2 = px.bar(df, x="Location", y="Pollution_PM2.5", color="Location")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔗 Traffic vs Pollution")
    fig3 = px.scatter(
        df,
        x="Traffic_Count",
        y="Pollution_PM2.5",
        color="Location",
        size="Traffic_Count",
        hover_data=["Time", "Weather", "Avg_Speed"]
    )
    st.plotly_chart(fig3, use_container_width=True)

    documents = []

    for _, row in df.iterrows():
        text = (
            f"At {row['Time']} in {row['Location']}, "
            f"traffic count was {row['Traffic_Count']} vehicles, "
            f"pollution level was {row['Pollution_PM2.5']} PM2.5, "
            f"average speed was {row['Avg_Speed']} km/h, "
            f"and weather was {row['Weather']}."
        )
        documents.append(text)

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

        if "highest traffic" in q or "most congested" in q or "maximum traffic" in q:
            row = df.loc[df["Traffic_Count"].idxmax()]
            st.write(
                f"The most congested area is **{row['Location']}** at **{row['Time']}** "
                f"with **{row['Traffic_Count']} vehicles**. "
                f"The pollution level there is **{row['Pollution_PM2.5']} PM2.5**."
            )

        elif "highest pollution" in q or "most polluted" in q or "maximum pollution" in q:
            row = df.loc[df["Pollution_PM2.5"].idxmax()]
            st.write(
                f"The highest pollution is recorded at **{row['Location']}** at **{row['Time']}** "
                f"with PM2.5 level of **{row['Pollution_PM2.5']}**. "
                f"The traffic count at that time is **{row['Traffic_Count']} vehicles**."
            )

        elif "traffic affect pollution" in q or "traffic and pollution" in q or "relation" in q or "correlation" in q:
            correlation = df["Traffic_Count"].corr(df["Pollution_PM2.5"])
            st.write(
                f"The correlation between traffic count and PM2.5 pollution is **{correlation:.2f}**. "
                f"This shows that higher traffic is generally linked with higher pollution levels."
            )

        elif "suggest" in q or "reduce" in q or "solution" in q:
            st.write("""
            Suggestions to reduce congestion and pollution:

            1. Improve traffic signal timing during peak hours.
            2. Promote public transport and carpooling.
            3. Restrict heavy vehicles in crowded zones during peak time.
            4. Use real-time traffic monitoring for route planning.
            5. Increase green zones near high-pollution areas.
            """)

        elif "summary" in q:
            st.write(
                "The dataset shows traffic and pollution levels across different city areas. "
                "High traffic areas such as City Center, Bus Stand, and Industrial Area show higher PM2.5 levels. "
                "Lower traffic areas such as Residential Area show lower pollution levels."
            )

        else:
            st.write("Based on the retrieved smart city data, these records are most relevant to your question:")
            for item in retrieved_data:
                st.write(item)

        with st.expander("Retrieved RAG Context"):
            for item in retrieved_data:
                st.write(item)

else:
    st.info("Please upload the traffic_data.csv file to start.")
