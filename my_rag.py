import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import PyPDF2
import hashlib
from pathlib import Path
import textwrap

os.makedirs('uploads', exist_ok=True)

class RAGConfig:
    def __init__(self):
        self.google_api_key = ""
        self.embedding_model = "all-MiniLM-L6-v2"
        self.vector_dimension = 384
        self.top_k_results = 5
        self.index_path = "faiss_index.idx"
        self.processed_files_path = "processed_files.pkl"
        self.document_store_path = "document_store.pkl"
        self.upload_directory = "uploads"

class Document:
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}
        self.embedding = None

class PDFProcessor:
    @staticmethod
    def process_pdf(file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return ""

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class RAGSysstem:
    def __init__(self, api_key=None):
        self.config = RAGConfig()
        if api_key:
            self.config.google_api_key = api_key

        genai.configure(api_key=self.config.google_api_key)
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        self.documents = []
        self.index = None
        self.processed_files = set()
        self._load_or_create_resources()
        print("RAG system initialized successfully!")

    def _load_or_create_resources(self):
        try:
            if os.path.exists(self.config.document_store_path):
                with open(self.config.document_store_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Loaded {len(self.documents)} documents from store")

            if os.path.exists(self.config.index_path):
                self.index = faiss.read_index(self.config.index_path)
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                self._create_index()

            if os.path.exists(self.config.processed_files_path):
                with open(self.config.processed_files_path, 'rb') as f:
                    self.processed_files = pickle.load(f)
                print(f"Loaded {len(self.processed_files)} processed file hashes")
        except Exception as e:
            print(f"Error loading resources: {e}")
            self.documents = []
            self.processed_files = set()
            self._create_index()

    def _create_index(self):
        self.index = faiss.IndexFlatL2(self.config.vector_dimension)
        print("Created new FAISS index")

    def _save_resources(self):
        with open(self.config.document_store_path, 'wb') as f:
            pickle.dump(self.documents, f)

        faiss.write_index(self.index, self.config.index_path)

        with open(self.config.processed_files_path, 'wb') as f:
            pickle.dump(self.processed_files, f)

    def _embed_text(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text)

    def ingest_document(self, text: str, metadata: Dict[str, Any] = None) -> None:
        chunks = self._chunk_text(text)
        for chunk in chunks:
            doc = Document(chunk, metadata)
            doc.embedding = self._embed_text(chunk)
            self.documents.append(doc)

            if self.index.ntotal == 0:
                self.index = faiss.IndexFlatL2(doc.embedding.shape[0])
            self.index.add(np.array([doc.embedding], dtype=np.float32))

        self._save_resources()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        text = text.replace('\n', ' ').strip()
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 200:
                chunks.append(chunk)
        return chunks

    def upload_pdf(self, file_path: str) -> bool:
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False

            if Path(file_path).suffix.lower() != '.pdf':
                print("Only PDF files are supported.")
                return False

            file_hash = PDFProcessor.get_file_hash(file_path)

            if file_hash in self.processed_files:
                print(f"File already processed: {file_path}")
                return True

            text = PDFProcessor.process_pdf(file_path)

            if not text:
                print("Failed to extract text.")
                return False

            metadata = {
                "source": os.path.basename(file_path),
                "path": file_path,
                "hash": file_hash
            }

            print(f"Processing {file_path}...")
            self.ingest_document(text, metadata)

            self.processed_files.add(file_hash)
            self._save_resources()

            print("✅ Successfully processed.")
            return True

        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return False

    def retrieve(self, query: str) -> List[Document]:
        if not self.documents or self.index.ntotal == 0:
            return []

        query_embedding = self._embed_text(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        distances, indices = self.index.search(query_embedding, min(self.config.top_k_results, self.index.ntotal))
        return [self.documents[i] for i in indices[0] if i != -1]

    def ask(self, query: str, show_sources: bool = True) -> str:
        if not self.documents:
            return "No documents uploaded."

        print(f"Q: {query}")
        docs = self.retrieve(query)

        if not docs:
            response = self._generate_with_llm(query, "")
            return response

        context = "\n\n".join([doc.text for doc in docs])
        response = self._generate_with_llm(query, context)

        if show_sources:
            sources = list({doc.metadata.get("source") for doc in docs if "source" in doc.metadata})
            sources_text = "\n\nSources: " + ", ".join(sources) if sources else ""
            return response + sources_text

        return response

    def _generate_with_llm(self, query: str, context: str) -> str:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )

        prompt = f"""
        Use the context below to answer the question. If the context doesn't help, say so.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """ if context else query

        response = model.generate_content(prompt)
        return response.text
