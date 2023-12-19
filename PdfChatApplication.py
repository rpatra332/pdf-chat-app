import os
import streamlit as st
from streamlit_chat import message
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain

from util import get_raw_text_from_pdf, split_raw_text


page_icon = "📕"
layout = "centered"
page_title = "Pdf Chat"
caption_text = "By <a href=\"https://github.com/rpatra332\" target=\"_blank\" rel=\"noopener noreferrer\" class=\"mycaption\">Rohit Patra</a>"

st.set_page_config(page_icon=page_icon,
                   page_title=page_title, layout=layout)
st.title(body=f'📕 PDF Chat',
         help="Made With LangChain And Google Gemini API")
st.caption(caption_text, unsafe_allow_html=True)

# --- PDF FROM UI ---
with st.form(key="form"):
    pdf_file = st.file_uploader(label="Upload the PDF you want to have a chat with.",
                                accept_multiple_files=False, type='pdf', )
    upload = st.form_submit_button(label="Upload")

_GOOGLE_GENERATIVE_API_KEY = os.environ['GOOGLE_GENERATIVE_API_KEY']

if pdf_file:
    # --- PDF TO TEXT ---
    raw_text = get_raw_text_from_pdf(pdf_file)

    # --- RAW TEXT TO SMALL CHUNKS ---
    text_chunks = split_raw_text(raw_text=raw_text)

    # --- EMBEDDING DOWNLOAD ---
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=_GOOGLE_GENERATIVE_API_KEY,
        model="models/embedding-001"
    )

    # --- BUILDING FAISS VECTOR STORE ---
    document_search = FAISS.from_texts(text_chunks, embeddings)

    # --- GEMINI-PRO QA CHAIN ---
    GeminiPro = ChatGoogleGenerativeAI(
        google_api_key=_GOOGLE_GENERATIVE_API_KEY,
        model="gemini-pro",
        convert_system_message_to_human=True
    )
    chain = load_qa_chain(GeminiPro, chain_type="stuff")

    # --- CREATING A STATE TO STORE CHAT HISTORY ---
    if "messages" not in st.session_state:
        st.session_state.messages = list()

    # --- QUERY INPUT UI ---
    with st.form("query-form"):
        query = st.text_input(label="Enter Your Query.")
        submit = st.form_submit_button(label="Submit")

    # --- QUERY RESULT AND HISTORY UI ---
    if submit and query:
        st.subheader("Query Result And History:")
        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append(
            {"role": "assistant", "content": result})

        with st.container(border=True):
            for i, msg in reversed(list(enumerate(st.session_state.messages))):
                message(msg["content"],
                        is_user=msg["role"] == "user",
                        key=f"{msg['role']}_{i}",
                        )


# --- CSS PROPERTIES ---
caption_css_change = """
<style>
    .mycaption{
        color: rgba(250, 250, 250, 0.8) !important;
        text-decoration: none;
    }
</style>
"""
st.markdown(caption_css_change, unsafe_allow_html=True)
