from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter,TokenTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

class RAG:
    def __init__(self) -> None:
        self.pdf_folder_path = os.getenv('SOURCE_DATA')
        self.emb_model_path = os.getenv('EMBED_MODEL')
        self.emb_model = self.get_embedding_model(self.emb_model_path)
        self.vector_store_path = os.getenv('VECTOR_STORE')

    def load_docs(self,path:str) -> PyPDFDirectoryLoader:
        loader = PyPDFDirectoryLoader(path)
        docs = loader.load()
        return docs
    
    def get_embedding_model(self,emb_model) -> HuggingFaceBgeEmbeddings :
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        embeddings_model = HuggingFaceBgeEmbeddings(
            model_name=emb_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embeddings_model
    
    def split_docs(self,docs)-> TokenTextSplitter:
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        return documents
    
    def populate_vector_db(self) -> None:
        # load embeddings into Chroma - need to pass docs , embedding function and path of the db
        for filename in os.listdir(self.vector_store_path):
            file_path = os.path.join(self.vector_store_path, filename)
            try:
                # If it's a file or a symbolic link, remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                # If it's a directory, remove it and all its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Deleted directory: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        self.doc = self.load_docs(self.pdf_folder_path)
        self.documents = self.split_docs(self.doc)
        
        db = Chroma.from_documents(self.documents,
                                   embedding=self.emb_model,
                                   persist_directory=self.vector_store_path)
        
        db.persist()
    
    def load_vector_db(self)-> Chroma:
        #to load back the embeddings from disk 
        db = Chroma(persist_directory=self.vector_store_path,embedding_function=self.emb_model)
        return db
    
    def get_retriever(self) -> Chroma:
        return self.load_vector_db().as_retriever()


