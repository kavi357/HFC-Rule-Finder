# HFC Rule Finder Version 05
import os
import streamlit as st
import pickle
import time
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore


st.markdown("""
    <style>
    /* Style for the button */
    .stButton > button {
        border: none;
        background-color: transparent;
        color: white;
        padding: 10px 16px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s;
    }

    /* Change color on hover */
    .stButton > button:hover {
        background-color: transparent;
        color: red !important;    
    }

    /* Force orange color when clicked */
    .stButton > button:active {
       background-color: transparent;
       color: red  !important;   
    }

    /* Make sure it stays orange if focused */
    .stButton > button:focus {
        background-color: transparent;
        color: red !important;
        outline: none;
        box-shadow: none;
    }
    </style>
""", unsafe_allow_html=True)

############################################# Functions ####################################################

# manual function to fetch web page content
def fetch_webpage_content(url):
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request failed
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=" ")
        return text.strip()
    except Exception as e:
        print(f"Error fetching or processing {url}, exception: {e}")
        return None


# summarize function
def summarize():
    main_placeholder.text("Rule Finder is in action!  Collecting your data... Stay tuned! 🕵️‍♂️🕵️‍♂️🕵️‍♂️")
    time.sleep(2)
    loaded_texts = []
    content = fetch_webpage_content(url)

    if content:
        loaded_texts.append({"text": content, "metadata": {"source": url}})
    else :
        main_placeholder.text("Oops! Looks like the URL didn't go through. Please enter it and try again. ⚠️⚠️⚠️")      
        return  

    main_placeholder.text("Data loading complete! Rule Finder is ready for action! 🏆🏆🏆")
    time.sleep(2)        

    # Proceed if we have loaded texts
    if loaded_texts:
        print('\n\n\nshow summary clicked \n\n\n')
        main_placeholder.text("Pipeline setup activated! Rule Finder is on the move! 🚨🚨🚨")
        time.sleep(2)
        summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
        st.header("Summary of Loaded Documents")

        main_placeholder.text("Pipeline built! 🛠️ Rule Finder is ready for the next phase! 🎯🎯🎯")
        time.sleep(2)
        
        # Combine all loaded texts
        all_text = " ".join([entry['text'] for entry in loaded_texts])
        # all_text = loaded_texts['text'] 
        
        main_placeholder.text("Exploring whether chunks are wanted! 🔍🔍🔍")
        time.sleep(2)

        main_placeholder.text(f"Docunment length is {len(all_text)}")
        time.sleep(2)

        # Break down the text into chunks if it's too long for the summarization model
        chunk_size = 3000  # Size for splitting the text into chunks
        if len(all_text) > chunk_size:
            chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]
            main_placeholder.text("Need chunks! Let’s make some! 🌟🌟🌟")
            time.sleep(2)
        else:
            chunks = [all_text]
            main_placeholder.text("Chunks not required! Let's go! 🏃‍♂️🏃‍♂️🏃‍♂️")
            time.sleep(2)
        
        # Generate summaries for each chunk
        main_placeholder.text("Generating chunk overviews! 🧩 This might take a bit! 🕒🕒🕒 ")
        time.sleep(2)
        summaries = []
        for chunk in chunks:
            try:
                summary = summarization_pipeline(chunk)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
        
        main_placeholder.text("Rule Finder is finalizing the summary now! 🕒 Almost done! 🕵️‍♂️🕵️‍♂️🕵️‍♂️ ")
        time.sleep(2)
        
        # Combine the summaries
        main_placeholder.text("All done and dusted! Time to unveil the masterpiece: the summary! 💎💎💎")
        time.sleep(2)
        final_summary = " ".join(summaries)
        
        # Display the generated summary
        st.write(final_summary)
        print('\n\n final summary\n\n')
        print(final_summary)
    else:
        print("No texts were loaded for summarization.")


def process_url() :
    # load data
    print(f'\n\n\n\n\nStep 01 - Url {st.session_state.urls}\n\n\n\n')
    # loader = UnstructuredURLLoader(urls=urls)
    # main_placeholder.text("Data Loading...Started...")
    # data = loader.load() not using unstructured url loader


    loaded_texts = []
    for url in st.session_state.urls:
        content = fetch_webpage_content(url)
        if content:
            loaded_texts.append({"text": content, "metadata": {"source": url}})
        else :
            main_placeholder.text("Oops! Looks like the URL didn't go through. Please enter it and try again. ⚠️⚠️⚠️")      
            return      

    print(f'\n\n\n\n\nStep 02 - Data Loading \n\n\n\n')
    print(f'\n\n\n\n\n{loaded_texts}\n\n\n\n\n')
    main_placeholder.text("Data in Place! The Rulr Finder is Ready to process!...📂📂📁")
    time.sleep(2)

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    # main_placeholder.text("Text Splitter...Started...")

    # Split the documents and retain the metadata
    docs = []
    for entry in loaded_texts:
        # Split each document text
        split_docs = text_splitter.create_documents([entry["text"]])
        # Add metadata to each split document
        for doc in split_docs:
            doc.metadata = entry["metadata"]
        docs.extend(split_docs)

    print(f'\n\n\n\n\nStep 03 - Data Splitted \n\n\n\n')
    print(f'\n\n\n\n\n{docs}\n\n\n\n\n')
    main_placeholder.text("The Rule Finder Has Split the Text! 🧩🧩🧩")
    time.sleep(2)

    print(f'\n\n\n\n\nStep 04 - Start Creating Embeddings \n\n\n\n')
    # time.sleep(2)

    # create embeddings using hugging face 
    # main_placeholder.text("Embedding Vector Started Building...")
    # Use Hugging Face SentenceTransformer for embeddings
    hf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Create embeddings for each document
    doc_texts = [doc.page_content for doc in docs]
    embeddings = hf_model.encode(doc_texts, convert_to_tensor=True)

    print(f'\n\n\n\n\nStep 05 - Hugging Face Embeddings Created\n\n\n\n')
    print(f'{embeddings}')
    print(f'\n\n\n\n\n\n')
    # main_placeholder.text("The Parrot Has Created the Embedding Vector! 📗📗📗")

    print(f'\n\n\n\n\nStep 06 - Save Embeddings \n\n\n\n')
    print(f'\n\n\n\n\n\n\n\n\n\n')

    main_placeholder.text("Embedding Start Saving...")
    # Create FAISS index
    dimension = embeddings.shape[1]  # Length of embedding vectors
    index = faiss.IndexFlatL2(dimension)
    # Add the embeddings to the FAISS index
    index.add(np.array(embeddings))
    # Save FAISS index to disk
    faiss.write_index(index, "faiss_store.index")
    # Save the documents (metadata)
    with open("docs_metadata.pkl", "wb") as f:
        pickle.dump(docs, f)
    main_placeholder.text("The Rule Finder Has Created the Embedding Vector! 🎯🎯🎯")    
    time.sleep(2)
    main_placeholder.text("The Rule Finder is Ready!! ⚡⚡⚡") 
    time.sleep(2)

    print(f'\n\n\n\n\nStep 07 - Faiss index Created\n\n\n\n')
    print(f'\n\n\n\n\n{index}\n\n\n\n\n')
    print(f'\n\n\n\n\nStep 08 - Saved Embeddings Successfully\n\n\n\n')

    print(f'\n\n\n\n\nStep 09 - Verifying faiss index\n\n\n\n')
    index = faiss.read_index("faiss_store.index")
    print(f"Number of vectors in the index: {index.ntotal}")
    # Convert query string into an embedding vector using the same model that created the embeddings
    query_text = "Tata Motors"
    query_vector = hf_model.encode([query_text], convert_to_tensor=False)

    # Ensure the query_vector is in the correct shape (NumPy array)
    query_vector = np.array(query_vector)

    # Perform the search in FAISS index
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_vector, k=k)

    # Print the results
    print(f"Top 5 closest vectors indices: {indices}")
    print(f"Distances: {distances}")

    print(f"Query vector shape: {query_vector.shape}")
    print(f"FAISS index dimension: {dimension}")

    ##### get decoded text

    # Get the indices of the top 5 closest vectors
    top_k_indices = indices[0]  # Since the result is a 2D array, take the first row

    # Retrieve the corresponding documents based on the indices
    top_k_docs = [docs[i] for i in top_k_indices]

    # Display the top 5 closest documents
    for i, doc in enumerate(top_k_docs):
        print(f"Document {i+1} (Index {top_k_indices[i]}):\n{doc.page_content}\n")

    ####
    print(f'\n\n\n\n\nVerifying faiss index finished\n\n\n\n')

def make_answer() :
    if os.path.exists("faiss_store.index"):
        # Load FAISS index and document metadata
        index = faiss.read_index("faiss_store.index")
        with open('docs_metadata.pkl', 'rb') as f:
            docs = pickle.load(f)

        place_holder.text("The Rule Finder is Gathering his Papers... Loading!... 🚀🚀🚀")
        time.sleep(2)    

        # Check if the number of docs matches the FAISS index size
        if len(docs) != index.ntotal:
            raise ValueError(f"Mismatch between number of documents ({len(docs)}) and FAISS index entries ({index.ntotal})")

        # Initialize the embeddings model
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Create a mapping from FAISS index to document store ID
        index_to_docstore_id = {i: str(i) for i in range(len(docs))}

        # Create an InMemoryDocstore to manage the documents
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

        # Create FAISS vectorstore with docstore and embeddings
        vectorstore = FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id)

        # Create a retriever
        retriever = vectorstore.as_retriever()

        # Define the LLM model for QA
        from transformers import pipeline
        qa_pipeline = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

        # Retrieve relevant documents from the vectorstore
        retrieved_docs = retriever.get_relevant_documents(query)

        place_holder.text("Mission Complete! The Rule Finder Retrieved the Relevant Docs!... 🎯🎯🎯")
        time.sleep(2)

        # Prepare the context (combine retrieved docs into one context for the pipeline)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # Ensure that the input is passed as a dictionary to the pipeline
        result = qa_pipeline({
            'question': query,
            'context': context
        })
        place_holder.text("The Rule Finder Finished the Job! Everything’s All Set!... 💎💎💎")
        time.sleep(2)
        
        # Extract the answer
        # st.header("Answer")
        st.header("The Rule Finder Speaks!")
        st.write(result['answer'])

        # Display sources
        sources = "\n".join([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs])
        if sources:
            st.subheader("Sources:")
            st.write(sources.split('\n')[0])    

        
###########################################################################################################


########################################### Rule Finder Main Program #########################################


# Custom CSS to remove button outline and focus styles
st.markdown("""
    <style>
    /* Remove button outline and focus */
    .stButton > button {
        border: none;
        background-color: transparent;
        box-shadow: none;
    }
    .stButton > button:focus {
        outline: none;
        box-shadow: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("HFC Rule Finder 🕵️‍♂️💬")
st.subheader("The Rule Finder clarifies MMA rules, turning confusion into understanding! 🥋💡")
st.sidebar.title("Share URLs to Rule Finder! 🕵️‍♂️🔗")


# Initialize the session state for the URLs list
if "urls" not in st.session_state:
    st.session_state.urls = [""]

# Function to add a new input box
def add_input_box():
    st.session_state.urls.append("")  # Add a new empty string to the list

# Function to process URLs
def process_urls():
    # Placeholder function to process the URLs
    st.write("Processing the following URLs:")
    for url in st.session_state.urls:
        st.write(url)
        
fruits = ['🥊','🥋','📋','🌐','🔗']
# Display all the input boxes dynamically
for i, url in enumerate(st.session_state.urls):
    # st.session_state.urls[i] = st.sidebar.text_input(f"URL {i+1} for parrot", value=url, key=f"url_input_{i+1}")
    st.session_state.urls[i] = st.sidebar.text_input(f"give url {i+1} to the rule finder {fruits[i]}", value=url, key=f"url_input_{i+1}")


# Plus button to add more input boxes (without outline)
st.sidebar.button("➕", on_click=add_input_box)

# Button to process the URLs
main_placeholder = st.empty()
place_holder = st.empty()
query = None
process_url_clicked = st.sidebar.button("Send the URLs to Rule Finder 🕵️‍♂️🛠️")
file_path = "faiss_store.pkl"




#############################################################################################################

if process_url_clicked:
    process_url()
    print(st.session_state.urls)
    main_placeholder = st.empty()
# query = main_placeholder.text_input("Question: ")
query = main_placeholder.text_input("Query The Rule Finder! 🕵️‍♂️💬")
summarize_url_clicked = st.sidebar.button("Rule Finder, Make It Brief! 🕵️‍♂️🚀")

if summarize_url_clicked :
    summarize() 

if query :
    print('IN QUERY',query)
    make_answer()  
else :
    print('no quary',query) 

# file_path = "faiss_store_openai.pkl"
