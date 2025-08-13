import os
import streamlit as st
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import PromptTemplate

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------- Constants ----------
FAISS_INDEX_PATH = "faiss_index"
CSV_PATH = "Flipkart_FAQs.csv"

# ---------- Environment Setup ----------
try:
    load_dotenv()
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
    logger.info("Environment variables loaded.")
except Exception as e:
    logger.error(f"Failed to load environment variables: {e}")

# ---------- Standalone Functions ----------
def get_prompt_template() -> str:
    """
    Returns a templated prompt for generating a response using LLM.
    """
    return """
    You are a world class business development representative. 
    Respond to the customer's message below using best practices.

    Customer Message:
    {message}

    Best Practices:
    {best_practice}

    Provide the most appropriate and professional response.
    """

def build_context(conversation, new_message):
    """
    Build a conversation context string from chat history and new user message.

    Args:
        conversation (list): Chat history with roles and messages.
        new_message (str): New message from the user.

    Returns:
        str: Complete conversation context as string.
    """
    context = ""
    for entry in conversation:
        context += f"{entry['role']}: {entry['content']}\n"
    return context + f"user: {new_message}\n"

# ---------- Vector Store Handler ----------
class VectorStore:
    """
    Manages loading, saving, and querying the FAISS vector database.
    """

    def __init__(self, index_path=FAISS_INDEX_PATH, csv_path=CSV_PATH):
        self.index_path = index_path
        self.csv_path = csv_path
        self.embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
        self.db = self.load_or_create()

    def load_or_create(self):
        try:
            if os.path.exists(self.index_path):
                db = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
                st.info("Loaded FAISS index.")
                logger.info("Loaded FAISS index from disk.")
            else:
                docs = CSVLoader(file_path=self.csv_path).load()
                db = FAISS.from_documents(docs, self.embeddings)
                db.save_local(self.index_path)
                st.info("Created and saved new FAISS index.")
                logger.info("Created and saved new FAISS index.")
            return db
        except Exception as e:
            logger.error(f"Error loading or creating FAISS index: {e}")
            st.error(f"Error loading or creating FAISS index: {e}")
            return None

    def search(self, query, k=3):
        try:
            if not self.db:
                logger.warning("Vector store is not initialized.")
                return []
            return [doc.page_content for doc in self.db.similarity_search(query, k=k)]
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            st.error(f"Error during vector search: {e}")
            return []

# ---------- Chat Generator ----------
class ChatResponder:
    """
    Uses Cohere LLM to generate contextual responses.
    """

    def __init__(self):
        try:
            self.llm = ChatCohere(model="command-r", temperature=0.7)
            self.prompt = PromptTemplate(
                input_variables=["message", "best_practice"],
                template=get_prompt_template()
            )
            self.chain = self.prompt | self.llm
            logger.info("ChatResponder initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatResponder: {e}")
            self.chain = None

    def respond(self, message, best_practice):
        if not self.chain:
            logger.warning("LLM chain is not initialized.")
            return "Assistant is not available."
        try:
            input_data = {"message": message, "best_practice": best_practice}
            result = self.chain.invoke(input=input_data)
            return result.content if hasattr(result, 'content') else result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            st.error(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response."

# ---------- Conversation Handler ----------
class ConversationManager:
    """
    Manages chat history and display.
    """

    def __init__(self):
        if "conversation_history" not in st.session_state:
            st.session_state["conversation_history"] = []

    def add(self, user_msg, bot_msg):
        st.session_state["conversation_history"].append({"role": "user", "content": user_msg})
        st.session_state["conversation_history"].append({"role": "assistant", "content": bot_msg})

    def get_history(self):
        return st.session_state["conversation_history"]

    def display(self):
        for i in range(len(st.session_state["conversation_history"]) - 1, -1, -2):
            if i - 1 >= 0:
                user_msg = st.session_state["conversation_history"][i - 1]["content"]
                bot_msg = st.session_state["conversation_history"][i]["content"]
                st.info(f"**User:** {user_msg}")
                st.write("------------------------------------------------")
                st.success(f"**Assistant:** {bot_msg}")

# ---------- Main Streamlit App ----------
def run_app():
    st.set_page_config(page_title="CustomerCareBot", page_icon="🤖")
    st.title("CustomerCareBot 🤖")

    vector_store = VectorStore()
    responder = ChatResponder()
    conversation = ConversationManager()

    user_input = st.text_area("Enter customer message:", key="user_input")

    if user_input:
        with st.spinner("Generating response..."):
            context = build_context(conversation.get_history(), user_input)
            best_practices = vector_store.search(context)
            reply = responder.respond(context, best_practices)
            conversation.add(user_input, reply)
            st.session_state["user_input"] = ""
            logger.info(f"User: {user_input}\nAssistant: {reply}")

    conversation.display()

# ---------- Entrypoint ----------
if __name__ == "__main__":
    run_app()