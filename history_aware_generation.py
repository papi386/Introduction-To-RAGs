from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma #local vectorial db
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

persistent_directory='db/chroma_db'

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db=Chroma(
   
        persist_directory=persistent_directory,
         embedding_function=embedding_model, 
        collection_metadata={"hnsw:space": "cosine"}
)
def init_gemini_flash(api_key):
   # """Query Gemini Flash model"""
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        return model
        
    
    except Exception as e:
        print(f"Error initializing Gemini Flash: {e}")
        return None
def query_gemini_flash(prompt,model):
   # """Query Gemini Flash model"""
    try:
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        print(f"Error querying Gemini Flash: {e}")
        return None



chat_history=[]

def ask_question(user_question,model):
    print(f"\n--- You asked: {user_question} ---")
    
    # Build context from chat history (Q&A pairs)
    context = ""
    if chat_history:
        context = "Conversation history:\n"
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        context += "\n"

    # Step 1: (Optional) Rewrite question with context — Gemini can do this in one shot
    if chat_history:
        rewrite_prompt = (
            "Given the conversation history below, rewrite the following new question as a clear, standalone question "
            "that includes all necessary context. Only output the rewritten question.\n\n"
            f"{context}New question: {user_question}"
        )
        rewritten_response = query_gemini_flash(rewrite_prompt, model)
        if rewritten_response:
            search_question = rewritten_response.strip()
            print(f"Searching for: {search_question}")
        else:
            search_question = user_question
    else:
        search_question = user_question

    # Step 2: Retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Build final prompt with context + documents + question
    doc_texts = "\n".join([f"- {doc.page_content}" for doc in docs])
    final_prompt = (
        f"{context}"
        f"Answer the following question using ONLY the information from these provided documents. "
        f"If the answer cannot be determined from the documents, say: "
        f"'I don't have enough information to answer that question based on the provided documents.'\n\n"
        f"Documents:\n{doc_texts}\n\n"
        f"Question: {user_question}\n\n"
        f"Answer:"
    )

    # Step 4: Query Gemini
    answer = query_gemini_flash(final_prompt, model)
    if answer is None:
        answer = "Sorry, I encountered an error while generating a response."

    # Step 5: Save to chat history (as simple dict with role/content)
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"Answer: {answer}")
    return answer


def start_chat(model):
    print("Asm me questions!Type 'quit' to exit.")
    while(True):
        question=input("\nYour Question:")
        if (question.lower()=='quit'):
            print("BYE !")
            break
        ask_question(question,model)


def main():
    print("MAIN")
    #1. initializing the model
    api_key = os.getenv("gemini")
    model=init_gemini_flash(api_key)
    #2. start the conversation
    start_chat(model)


if __name__=='__main__':
    main()