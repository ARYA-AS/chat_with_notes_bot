import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline

# ------------------- Streamlit UI Setup -------------------
st.set_page_config(page_title="Chat with Notes", page_icon="📄")

st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .css-1aumxhk {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 CHAT WITH YOUR NOTES ")
st.markdown("Upload your academic PDF notes and ask questions.🚀")

with st.sidebar:
    st.header("📚 Instructions")
    st.markdown("""
    - Upload a clear academic PDF  
    - Wait a few seconds (processed once only)  
    - Ask any question related to the content  
    - Uses **Flan-T5-small** model for fast replies  
    """)

# ------------------- PDF Processing -------------------
@st.cache_resource(show_spinner=False)
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text and len(text.strip()) > 50:
            raw_text += text

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    return db

# ------------------- Main Interface -------------------
pdf = st.file_uploader("### 📂 Upload Your Notes (PDF)", type="pdf")

if pdf:
    with st.spinner("⏳ Processing your file... Please wait..."):
        db = process_pdf(pdf)

    # Load lightweight HuggingFace LLM pipeline
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=128,
        temperature=0.5
    )
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    chain = load_qa_chain(llm, chain_type="stuff")

    st.success("✅ Done! You can now ask questions below.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat UI input
    user_input = st.chat_input("Ask a question based on your uploaded PDF...")

    if user_input:
        # Show user message
        st.chat_message("user").markdown(user_input)

        # Get answer
        docs = db.similarity_search(user_input)
        response = chain.run(input_documents=docs, question=user_input)

        # Show bot reply
        st.chat_message("assistant").markdown(response)

        # Store history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", response))
