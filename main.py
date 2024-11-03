import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# sys.path.append("..") # Adds higher directory to python modules path.
# print (os.getcwd())

from pathlib import Path
from typing import List
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from config.config import Config
from embeddings.phobert_embeddings import PhoBERTEmbedding
from rag.rag_pipeline import RAGPipeline
import warnings

def read_pdf_files(pdf_dir: str) -> List[str]:
    """Đọc tất cả file PDF trong thư mục và trả về nội dung text"""
    documents = []
    pdf_path = Path(pdf_dir)
    
    for pdf_file in pdf_path.glob("*.pdf"):
        reader = PdfReader(str(pdf_file))
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        documents.append(text)
    
    return documents

def main():
    total_steps = 7  # Tổng số bước cần thực hiện
    current_step = 0
    
    def update_progress(message: str, step: int):
        nonlocal current_step
        current_step = step
        progress = (current_step / total_steps) * 100
        print(f"{message} ({progress:.0f}%)", end="\r")
    
    def complete_step(message: str):
        nonlocal current_step
        current_step += 1
        progress = (current_step / total_steps) * 100
        print(f"{message} ({progress:.0f}%)")
    
    print("Đang khởi tạo chương trình...")
    
    # Load environment variables
    update_progress("Đang tải biến môi trường...", 0)
    load_dotenv()
    complete_step("Đã tải biến môi trường thành công!")
    
    # Initialize config
    update_progress("Đang khởi tạo cấu hình...", 1)
    config = Config()
    complete_step("Đã khởi tạo cấu hình thành công!")
    
    # Initialize embedding model
    update_progress("Đang khởi tạo mô hình embedding...", 2)
    embedding_model = PhoBERTEmbedding(config)
    complete_step("Đã khởi tạo mô hình embedding thành công!")
    
    # Initialize RAG pipeline
    update_progress("Đang khởi tạo RAG pipeline...", 3)
    rag = RAGPipeline(config, embedding_model)
    complete_step("Đã khởi tạo RAG pipeline thành công!")
    
    # Read documents
    update_progress("Đang đọc tài liệu PDF...", 4)
    pdf_directory = "data/pdfs"  # Change this to your PDF folder path
    documents = read_pdf_files(pdf_directory)
    complete_step("Đã đọc tài liệu PDF thành công!")
    
    # Initialize vector store
    update_progress("Đang khởi tạo vector store...", 5)
    rag.initialize_vectorstore(documents)
    complete_step("Đã khởi tạo vector store thành công!")
    
    # Initialize QA chain
    update_progress("Đang khởi tạo QA chain...", 6)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    rag.initialize_qa_chain()
    complete_step("Đã khởi tạo QA chain thành công!")
    
    # Test query
    print("\nBắt đầu thử nghiệm truy vấn...")
    question = "Hãy cho tôi biết nội dung của tài liệu số 2?"
    answer = rag.query(question)
    print(f"Câu hỏi: {question}")
    print(f"Trả lời: {answer}")

if __name__ == "__main__":
    main() 