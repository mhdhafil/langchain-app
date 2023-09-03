from flask import Flask,request, flash, redirect, url_for
import os
import openai
import requests
import re
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders.base import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import validators
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)



openai.api_key = os.environ.get("OPENAI_API_KEY")




def docPrePro(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    documents = text_splitter.split_documents(doc)
    db = FAISS.from_documents(documents, OpenAIEmbeddings())

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        memory = memory,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain


# Function to scrape only visible text from the given URL
def scrape_visible_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove script, style, and other non-visible tags
    for tag in soup(["script", "style", "meta", "link", "noscript", "header", "footer", "aside", "nav", "img"]):
        tag.extract()

    # Get the header content
    header_content = soup.find("header")
    header_text = header_content.get_text() if header_content else ""

    # Get the paragraph content
    paragraph_content = soup.find_all("p")
    paragraph_text = " ".join([p.get_text() for p in paragraph_content])

    # Combine header and paragraph text
    visible_text = f"{header_text}\n\n{paragraph_text}"

    # Remove multiple whitespaces and newlines
    visible_text = re.sub(r'\s+', ' ', visible_text)
    return visible_text.strip()

#flask starting
app= Flask(__name__)

app.secret_key = "langchain"
qa = RetrievalQA


@app.route('/',methods=['POST','GET'])


@app.route('/submit_url', methods=['POST'])
def submit_url():
    global qa
    if request.method == 'POST':
        url = request.args.get('url')
        if validators.url(url):
            text = scrape_visible_text_from_url(url)
            d = []
            d.append(Document(page_content=text))
            qa = docPrePro(d)
            print("chat now")
            flash('You can chat now...')
            return redirect(url_for('send_msg'))



            
@app.route('/upload_document', methods=['POST'])
def upload_document():
    global qa
    
    uploaded_file = request.files['file']
    if uploaded_file:
        print("uploaded")
        destination = os.path.join('uploads/',uploaded_file.filename)
        print("data fetched")
        uploaded_file.save(destination)
        loader = PyPDFLoader(destination)
        d = loader.load()
        qa = docPrePro(d)
        print("chat now")
        flash('You can chat now...')
        os.remove(destination)
        return redirect(url_for('send_msg'))
    else:
        return {'data':'Unsuccessful'}

    # return {'data':'File uploaded successfully'}


@app.route('/send_msg', methods=['POST'])
def send_msg():
    user_msg = request.args.get('msg')
    result = qa({"query": user_msg})
    return result['result']


if __name__ == '__main__':
    app.run(debug=True)