import streamlit as st
import pandas as pd
import numpy as np
import faiss
import plotly.express as px
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Smart City RAG Assistant",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Smart City Intelligence Assistant")
st.write("RAG-based assistant for traffic, pollution, and smart city decision-making.")

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

def create_documents(df):
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

    return documents

def create_vector_store(documents):
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings

def retrieve_relevant_data(query, documents, index, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append(documents[i])

    return results

def generate_answer(query, df, retrieved_data):
    query = query.lower()

    if "highest traffic" in query or "most congested" in query:
        row = df.loc[df["Traffic_Count"].idxmax()]
        return (
            f"The most congested area is {row['Location']} at {row['Time']} "
            f"with {row['Traffic_Count']} vehicles. "
            f"The pollution level there is {row['Pollution_PM2.5']} PM2.5."
        )

    elif "highest pollution" in query or "most polluted" in query:
        row = df.loc[df["Pollution_PM2.5"].idxmax()]
        return (
            f"The highest pollution is recorded at {row['Location']} at {row['Time']} "
            f"with PM2.5 level of {row['Pollution_PM2.5']}. "
            f"The traffic count at that time is {row['Traffic_Count']} vehicles."
        )

    elif "traffic affect pollution" in query or "traffic and pollution" in query:
        correlation = df["Traffic_Count"].corr(df["Pollution_PM2.5"])

        return (
            f"Traffic and pollution are related. In this dataset, the correlation between "
            f"traffic count and PM2.5 pollution is {correlation:.2f}. "
            f"This means that as traffic increases, pollution also generally increases."
        )

    elif "suggest" in query or "reduce" in query or "solution" in query:
        return (
            "Suggestions to reduce traffic congestion and pollution:\n\n"
            "1. Improve traffic signal timing during peak hours.\n"
            "2. Promote public transport and carpooling.\n"
            "3. Restrict heavy vehicles in crowded zones during peak time.\n"
            "4. Use real-time traffic monitoring for route planning.\n"
            "5. Increase green zones near high-pollution areas."
        )

    elif "summary" in query:
        return (
            "Summary: The dataset shows traffic and pollution levels across different city areas. "
            "High traffic areas such as City Center, Bus Stand, and Industrial Area show higher PM2.5 levels. "
            "Lower traffic areas such as Residential Area show lower pollution levels."
        )

    else:
        return (
            "Based on the retrieved smart city data, these records are most relevant to your question:\n\n"
            + "\n\n".join(retrieved_data)
        )

uploaded_file = st.file_uploader("Upload traffic and pollution CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    documents = create_documents(df)
    index, embeddings = create_vector_store(documents)

    st.success("Data processed successfully!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Average Traffic", round(df["Traffic_Count"].mean(), 2))

    with col3:
        st.metric("Average PM2.5", round(df["Pollution_PM2.5"].mean(), 2))

    st.subheader("📊 Uploaded Data")
    st.dataframe(df)

    st.subheader("📈 Traffic Count by Location")
    fig1 = px.bar(df, x="Location", y="Traffic_Count", color="Location")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🌫️ Pollution Level by Location")
    fig2 = px.bar(df, x="Location", y="Pollution_PM2.5", color="Location")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔗 Traffic vs Pollution Relationship")
    fig3 = px.scatter(
        df,
        x="Traffic_Count",
        y="Pollution_PM2.5",
        color="Location",
        size="Traffic_Count",
        hover_data=["Time", "Avg_Speed", "Weather"]
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🤖 Ask the Smart City Assistant")

    query = st.text_input("Ask a question:")

    if query:
        retrieved_data = retrieve_relevant_data(query, documents, index)
        answer = generate_answer(query, df, retrieved_data)

        st.subheader("AI Answer")
        st.write(answer)

        with st.expander("Retrieved RAG Context"):
            for data in retrieved_data:
                st.write(data)

else:
    st.info("Upload traffic_data.csv to start.")
