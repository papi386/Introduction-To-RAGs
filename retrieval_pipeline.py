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

query="In what year was tunisia independent?"

#first way to call the retreiver:
retriever= db.as_retriever(search_kwargs={"k":5})

#second way :
"""
retriever= db.as_retriever(
    search_type="similarity_score_threshold" ,
    search_kwargs={
        "k":3,
        "score_threshold":0.3 #only return with cosine similarity >= 0.3
    }
)
"""

relevant_docs=retriever.invoke(query)


print("user query:{query}")

print("----Context----")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")


def query_gemini_flash(prompt, api_key):
   # """Query Gemini Flash model"""
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        print(f"Error querying Gemini Flash: {e}")
        return None

# Replace your previous LLM call with:
# Get your Google API key from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY =os.getenv("gemini")
while (True):
    query=input("Your Question:")
    # Your existing combined input
    relevant_docs=retriever.invoke(query)
    combined_input = f"""Based on the following documents, please answer this question: {query}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # Query Gemini Flash instead
    result = query_gemini_flash(combined_input, GOOGLE_API_KEY)

    if result:
        print("\n--- Generated Response ---")
        print("Content only:")
        print(result)
    else:
        print("Failed to get response from Gemini Flash")


