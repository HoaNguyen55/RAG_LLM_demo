from typing import List
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from embeddings.phobert_embeddings import PhoBERTEmbedding
from config.config import Config
import os
from transformers import BloomTokenizerFast, AutoTokenizer, AutoModel, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
import torch
from langchain.prompts import PromptTemplate

class LangchainPhoBERTEmbeddings(Embeddings):
    def __init__(self, phobert: PhoBERTEmbedding):
        self.phobert = phobert
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.phobert.get_embeddings(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.phobert.get_embeddings(text)
        return embedding.tolist()

class RAGPipeline:
    def __init__(self, config: Config, phobert: PhoBERTEmbedding, model_name="vinai/phobert-base"):
        self.config = config
        self.embeddings = LangchainPhoBERTEmbeddings(phobert)
        self.model_name = model_name
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", self.model_name.split('/')[-1])
        self.vectorstore = None
        self.qa_chain = None
        self.initialize_model()
        
    def initialize_model(self):
        """Kiểm tra và tải model nếu chưa có"""
        try:
            # Kiểm tra xem model đã được tải về chưa
            if os.path.exists(self.model_path):
                print(f"Đang tải model từ {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path)
            else:
                print(f"Tải model {self.model_name} từ Hugging Face")
                # Tạo thư mục models nếu chưa tồn tại
                os.makedirs(self.model_path, exist_ok=True)
                
                # Tải và lưu model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                
                # Lưu model và tokenizer
                self.tokenizer.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path)
                
            self.model.eval()  # Chuyển sang chế độ evaluation
            print("Model đã được tải thành công")
            
        except Exception as e:
            print(f"Lỗi khi tải model: {str(e)}")
            raise
        
    def initialize_vectorstore(self, documents: List[str]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        texts = text_splitter.split_text("\n\n".join(documents))
        
        self.vectorstore = FAISS.from_texts(
            texts,
            self.embeddings
        )
        
    def initialize_qa_chain(self):
        # Sử dụng phiên bản nhỏ hơn của PhoGPT để chạy trên CPU
        model_name = "vinai/PhoGPT-4B"  # Phiên bản nhẹ hơn 7B5
        
        # Cấu hình cho CPU
        config = {
            "torch_dtype": torch.float32,  # Dùng float32 thay vì float16
            "low_cpu_mem_usage": True,
            "device_map": "cpu"  # Chỉ định rõ sử dụng CPU
        }
        
        print("Đang tải PhoGPT model... (87%)")
        
        # Tải tokenizer và model
        tokenizer = BloomTokenizerFast.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        print("Đã tải tokenizer xong (88%)")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **config
        )
        print("Đã tải model xong (89%)")
        
        # Tạo pipeline với các tham số phù hợp cho CPU
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # -1 nghĩa là sử dụng CPU
            max_length=512,  # Giảm độ dài output để tiết kiệm bộ nhớ
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            num_return_sequences=1,
            batch_size=1  # Xử lý từng câu một
        )
        print("Đã tạo pipeline xong (90%)")
        
        # Tạo prompt template phù hợp
        template = """Dựa vào thông tin sau đây:
        {context}

        Hãy trả lời câu hỏi: {question}

        Trả lời ngắn gọn:"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Khởi tạo QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 2}  # Giảm số lượng document để tiết kiệm bộ nhớ
            ),
            chain_type_kwargs={
                "prompt": PROMPT
            },
            return_source_documents=True
        )
        
        print("Đã khởi tạo QA chain thành công! (91%)")
        
    def query(self, question: str) -> str:
        if self.qa_chain is None:
            raise ValueError("QA chain chưa được khởi tạo!")
        return self.qa_chain.run(question) 