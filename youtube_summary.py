import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from yt_dlp import YoutubeDL
from dotenv import load_dotenv
from langchain.schema import Document
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="LangChain Summarizer", page_icon="📝")
st.title("YouTube or Website Summarizer")
st.write("Welcome! Summarize content from YouTube videos or websites in a more detailed manner.")
st.sidebar.title("About This App")
st.sidebar.info(
    "This app uses LangChain and the Llama 3.2 model from Groq API to provide detailed summaries. "
    "Simply enter a URL (YouTube or website) and get a concise summary!"
)
st.header("How to Use:")
st.write("1. Enter the URL of a YouTube video or website you wish to summarize.")
st.write("2. Click **Summarize** to get a detailed summary.")
st.write("3. Enjoy the results!")
st.subheader("Enter the URL:")
generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="https://example.com")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
prompt_template = """
Provide a detailed summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def load_youtube_content(url):
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Video")
        description = info.get("description", "No description available.")
        return f"{title}\n\n{description}"

if st.button("Summarize"):
    if not generic_url.strip():
        st.error("Please provide a URL to proceed.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Processing..."):
                if "youtube.com" in generic_url:
                    text_content = load_youtube_content(generic_url)
                    docs = [Document(page_content=text_content)]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.subheader("Detailed Summary:")
                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception occurred: {e}")

