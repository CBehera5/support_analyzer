import os
import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import tempfile

# Set Streamlit page configuration
st.set_page_config(page_title="Support Ticket Analyzer", layout="wide")
st.title("🧠 Support Ticket Analyzer: Doc or UI Fix?")

# Sidebar for API Key input
groq_api_key = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password")

# File uploaders
ticket_file = st.file_uploader("📄 Upload Support Tickets CSV", type=["csv"])
doc_file = st.file_uploader("📘 Upload Documentation Text File", type=["txt"])

# Proceed only if all inputs are provided
if ticket_file and doc_file and groq_api_key:
    with st.spinner("Processing..."):

        # Save uploaded documentation to a temporary file
        temp_doc = tempfile.NamedTemporaryFile(delete=False)
        temp_doc.write(doc_file.read())
        temp_doc.close()

        # Load and split documentation
        loader = TextLoader(temp_doc.name)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        # Initialize vector store
        embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
        retriever = vectorstore.as_retriever(search_type="similarity", k=3)

        # Read support tickets
        df = pd.read_csv(ticket_file)
        df['combined'] = df['title'] + ". " + df['description']
        ticket_texts = df['combined'].tolist()

        # Initialize Groq client
        client = Groq(api_key=groq_api_key)

        # Process each ticket
        recommendations = []
        for ticket in ticket_texts:
            try:
                # Retrieve relevant documentation
                relevant_docs = retriever.get_relevant_documents(ticket)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Construct prompt
                prompt = f"""
You are an expert support analyst.

A user has reported a support ticket:
"{ticket}"

The following documentation was retrieved as relevant:
"{context}"

Based on this, determine:
- Is the user's problem due to unclear or missing documentation?
- Is the user struggling due to UI/UX issues?
- Is it both?

Respond in this format:
Category: [Update Documentation / Improve UI / Both]
Reason: [Brief explanation why]
"""

                # Call Groq API
                response = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3-8b-8192"
                )

                result = response.choices[0].message.content.strip()
                recommendations.append({"ticket": ticket, "recommendation": result})
            except Exception as e:
                recommendations.append({"ticket": ticket, "recommendation": f"Error: {e}"})

        # Display results
        result_df = pd.DataFrame(recommendations)
        st.success("✅ Analysis complete.")
        st.dataframe(result_df, use_container_width=True)

        # Download option
        csv_download = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", csv_download, "recommendations_output.csv", "text/csv")
else:
    st.info("👆 Please provide your Groq API Key and upload both the support tickets CSV and documentation file to proceed.")
