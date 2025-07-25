# ===============================
# 📦 Install required libraries
# ===============================
!pip install -q gradio google-generativeai sentence-transformers faiss-cpu PyPDF2

# ===============================
# 📂 Upload your `my_rag.py` file
# ===============================
from google.colab import files
uploaded = files.upload()  # Upload your my_rag.py here
# ===============================
# 🧠 Import your RAG system
# ===============================
from my_rag import RAGSysstem  # Make sure this matches the uploaded filename

# ===============================
# 🔐 Enter your Gemini API Key
# ===============================
import getpass
API_KEY = getpass.getpass("api key")

# ✅ Initialize RAG system
rag = RAGSysstem(api_key=API_KEY)

# ===============================
# 📘 Gradio chatbot functions
# ===============================
import gradio as gr

def upload_pdf_gradio(file):
    if file is None:
        return "No file uploaded."

    success = rag.upload_pdf(file.name)
    return "✅ File uploaded and processed." if success else "❌ Failed to process file."

def ask_question_gradio(question):
    if not question.strip():
        return "Please enter a valid question."

    return rag.ask(question)

# ===============================
# 🎨 Gradio UI
# ===============================
with gr.Blocks() as demo:
    gr.Markdown("# 📘 PDF Q&A Chatbot with Gemini")

    with gr.Row():
        file_input = gr.File(label="📎 Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("📤 Upload PDF")
    upload_output = gr.Textbox(label="Upload Status")
    upload_btn.click(upload_pdf_gradio, inputs=file_input, outputs=upload_output)

    gr.Markdown("---")

    question_box = gr.Textbox(label="❓ Ask a Question")
    answer_box = gr.Textbox(label="💬 Answer", lines=5)
    ask_btn = gr.Button("🔍 Ask")

    ask_btn.click(ask_question_gradio, inputs=question_box, outputs=answer_box)

# ===============================
# 🌐 Launch Gradio app
# ===============================
demo.launch(share=True)
