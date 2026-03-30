import os
from flask import Flask, request, jsonify, render_template_string
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

app = Flask(__name__)
vectorstore = None
chat_history = []

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Document Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: Arial, sans-serif; }
        body { background: #f5f5f5; padding: 20px; }
        h1 { color: #333; margin-bottom: 8px; }
        p.subtitle { color: #666; margin-bottom: 20px; }
        .container { display: flex; gap: 20px; max-width: 1200px; margin: 0 auto; }
        .left { width: 320px; flex-shrink: 0; }
        .right { flex: 1; }
        .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        label { display: block; font-weight: bold; margin-bottom: 8px; color: #444; }
        input[type=file] { width: 100%; padding: 10px; border: 2px dashed #ccc; border-radius: 8px; cursor: pointer; }
        button { width: 100%; padding: 12px; border: none; border-radius: 8px; background: #4F46E5; color: white; font-size: 15px; font-weight: bold; cursor: pointer; margin-top: 10px; }
        button:hover { background: #4338CA; }
        #status { margin-top: 12px; padding: 10px; border-radius: 8px; background: #f0fdf4; color: #166534; font-size: 14px; display: none; }
        .tips { margin-top: 16px; }
        .tips h3 { font-size: 14px; color: #555; margin-bottom: 8px; }
        .tip { background: #EEF2FF; color: #4F46E5; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 13px; cursor: pointer; }
        .tip:hover { background: #E0E7FF; }
        #chatbox { height: 420px; overflow-y: auto; padding: 16px; background: #fafafa; border-radius: 10px; border: 1px solid #eee; margin-bottom: 12px; }
        .msg { margin-bottom: 14px; }
        .msg.user .bubble { background: #4F46E5; color: white; margin-left: auto; }
        .msg.bot .bubble { background: white; border: 1px solid #eee; color: #333; }
        .bubble { max-width: 80%; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.6; display: inline-block; }
        .msg.user { text-align: right; }
        .input-row { display: flex; gap: 10px; }
        .input-row input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; }
        .input-row button { width: auto; padding: 12px 24px; margin: 0; }
        .clear-btn { background: #f3f4f6; color: #666; margin-top: 10px; font-size: 13px; padding: 8px; }
        .clear-btn:hover { background: #e5e7eb; }
        .thinking { color: #999; font-style: italic; font-size: 13px; }
    </style>
</head>
<body>
    <div style="max-width:1200px;margin:0 auto 20px;">
        <h1>AI Document Q&A Assistant</h1>
        <p class="subtitle">Upload any PDF or text document and ask questions in a natural conversation.</p>
    </div>
    <div class="container">
        <div class="left">
            <div class="card">
                <label>Upload Document (PDF or TXT)</label>
                <input type="file" id="fileInput" accept=".pdf,.txt">
                <button onclick="uploadDoc()">Load Document</button>
                <div id="status"></div>
            </div>
            <div class="card tips">
                <h3>Try asking:</h3>
                <div class="tip" onclick="setQ('What is this document about?')">What is this document about?</div>
                <div class="tip" onclick="setQ('Summarise the key points')">Summarise the key points</div>
                <div class="tip" onclick="setQ('What are the main exclusions?')">What are the main exclusions?</div>
                <div class="tip" onclick="setQ('How do I make a claim?')">How do I make a claim?</div>
                <div class="tip" onclick="setQ('What are the main conclusions?')">What are the main conclusions?</div>
            </div>
        </div>
        <div class="right">
            <div class="card" style="height:540px;display:flex;flex-direction:column;">
                <div id="chatbox">
                    <div class="msg bot"><div class="bubble">Hi! Upload a document and I will answer any questions about it. I remember our conversation so you can ask follow-up questions naturally.</div></div>
                </div>
                <div class="input-row">
                    <input type="text" id="qInput" placeholder="Ask a question about your document..." onkeydown="if(event.key==='Enter')sendQ()">
                    <button onclick="sendQ()">Ask</button>
                </div>
                <button class="clear-btn" onclick="clearChat()">Clear Conversation</button>
            </div>
        </div>
    </div>
<script>
function setQ(q) {
    document.getElementById('qInput').value = q;
    document.getElementById('qInput').focus();
}

async function uploadDoc() {
    const file = document.getElementById('fileInput').files[0];
    if (!file) { alert('Please select a file first.'); return; }
    const status = document.getElementById('status');
    status.style.display = 'block';
    status.textContent = 'Loading document...';
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    status.textContent = data.message;
    status.style.background = data.success ? '#f0fdf4' : '#fef2f2';
    status.style.color = data.success ? '#166534' : '#991b1b';
}

async function sendQ() {
    const input = document.getElementById('qInput');
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    addMsg('user', q);
    const thinking = addMsg('bot', '<span class="thinking">Thinking...</span>');
    const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q })
    });
    const data = await res.json();
    thinking.querySelector('.bubble').textContent = data.response;
}

function addMsg(role, html) {
    const box = document.getElementById('chatbox');
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.innerHTML = '<div class="bubble">' + html + '</div>';
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
}

async function clearChat() {
    await fetch('/clear', { method: 'POST' });
    const box = document.getElementById('chatbox');
    box.innerHTML = '<div class="msg bot"><div class="bubble">Conversation cleared. Ask me anything about your document.</div></div>';
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    global vectorstore, chat_history
    chat_history = []
    file = request.files.get("file")
    if not file:
        return jsonify({"success": False, "message": "No file received."})
    file_path = f"uploaded_{file.filename}"
    file.save(file_path)
    try:
        if file.filename.endswith(".pdf"):
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            documents = [Document(page_content=text, metadata={"source": file.filename})]
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return jsonify({"success": True, "message": f"Document '{file.filename}' loaded. {len(chunks)} sections indexed. You can now ask questions."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route("/chat", methods=["POST"])
def chat():
    global vectorstore, chat_history
    data = request.json
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "Please type a question."})
    if vectorstore is None:
        return jsonify({"response": "Please upload and load a document first."})
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(message)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        history_text = ""
        for human, ai in chat_history:
            history_text += f"User: {human}\nAssistant: {ai}\n"
        prompt_template = PromptTemplate.from_template("""You are an expert document analysis assistant.
Answer questions based only on the document context provided.
Be specific, accurate, and helpful.
If the answer is not in the document, clearly say so.

Document context:
{context}

Previous conversation:
{history}

Current question: {question}

Answer:""")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"context": context, "history": history_text, "question": message})
        chat_history.append((message, response))
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

@app.route("/clear", methods=["POST"])
def clear():
    global chat_history
    chat_history = []
    return jsonify({"success": True})

if __name__ == "__main__":
    print("Starting AI Document Assistant...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
