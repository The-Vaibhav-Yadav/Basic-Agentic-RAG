#! /usr/bin/env python

import os
import gradio as gr
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crewai import Crew, Task, Agent
from crewai.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

from create_agents import create_agents, create_tasks

load_dotenv()

# API Key Configuration
google_api_key = os.environ.get('GOOGLE_API_KEY')
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY required. Get it from https://makersuite.google.com/app/apikey")

tavily_api = os.environ.get('TAVILY_API_KEY')
if not tavily_api:
    raise ValueError("TAVILY_API_KEY required. Get it from https://tavily.com")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=tavily_api)

# Download PDF if not already present
def download_pdf():
    pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
    if not os.path.exists('attention_is_all_you_need.pdf'):
        print("Downloading PDF...")
        response = requests.get(pdf_url)
        with open('attention_is_all_you_need.pdf', 'wb') as file:
            file.write(response.content)
        print("PDF downloaded.")

# LLM Configuration - Using Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.1,
)

# Global vectorstore variable
vectorstore = None

# Initialize RAG vectorstore
def initialize_vectorstore():
    global vectorstore
    
    if vectorstore is not None:
        return vectorstore
    
    print("Initializing vectorstore...")
    
    # Load PDF
    loader = PyPDFLoader('attention_is_all_you_need.pdf')
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings - Using HuggingFace (not OpenAI)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"Vectorstore initialized with {len(splits)} document chunks.")
    return vectorstore

# Custom PDF Search Tool using CrewAI's @tool decorator
@tool("PDF Search Tool")
def pdf_search_tool(query: str) -> str:
    """Search the 'Attention is All You Need' paper for information about transformers, attention mechanisms, encoders, and decoders.
    Use this tool for questions about self-attention, multi-head attention, transformer architecture, or the attention paper."""
    vectorstore = initialize_vectorstore()
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant information found in the PDF."
    
    # Format results
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[Source {i}]\n{doc.page_content}")
    
    return "\n\n".join(results)

# Web Search Tool using CrewAI's @tool decorator
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Search the web for current information on any topic. Use this for general questions, recent developments, or topics not covered in the PDF."""
    try:
        response = tavily_client.search(query, max_results=3)
        
        if not response or 'results' not in response:
            return "No web results found."
        
        results = []
        for i, result in enumerate(response['results'], 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            url = result.get('url', '')
            results.append(f"[Result {i}] {title}\n{content}\nSource: {url}")
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Web search error: {str(e)}"

# Agent Definitions


# Main RAG Function with Fact Checking and Retry Logic
def run_rag_pipeline(question):
    try:
        # Download PDF if not exists
        download_pdf()
        
        # Initialize vectorstore
        initialize_vectorstore()

        # Setup tools
        tools = [pdf_search_tool, web_search_tool]

        # Retry logic variables
        max_retries = 2
        retries = 0
        previous_response = None
        
        while retries <= max_retries:
            print(f"\n{'='*60}")
            print(f"Attempt {retries + 1} of {max_retries + 1}")
            print(f"{'='*60}\n")
            
            # Modify question with retry context if needed
            current_question = question
            if retries > 0 and previous_response:
                current_question = (
                    f"IMPORTANT: This is attempt #{retries + 1}. "
                    f"The previous answer was INCORRECT according to fact checking.\n\n"
                    f"Previous incorrect response:\n{previous_response}\n\n"
                    f"Please research more carefully and provide a different, accurate answer.\n\n"
                    f"Original question: {question}"
                )
                print("⚠️  Retrying with corrected context...\n")

            # Create agents
            agents = create_agents()

            # Create tasks
            tasks = create_tasks(agents, tools)

            # Create Crew
            rag_crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=True,
            )

            # Run the pipeline
            result = rag_crew.kickoff(inputs={"question": current_question})
            
            # Get individual task outputs
            # Tasks order: research_task, fact_checking_task, writing_task
            tasks_output = rag_crew.tasks_output if hasattr(rag_crew, 'tasks_output') else None
            
            # Extract fact checker verdict and writer's answer
            fact_check_output = None
            writer_output = str(result)  # Final output is writer's response
            verdict_passed = True
            
            # Try to get fact checker output from tasks
            if tasks_output and len(tasks_output) >= 2:
                # Second task is fact checking
                fact_check_output = str(tasks_output[1].raw) if hasattr(tasks_output[1], 'raw') else str(tasks_output[1])
                
                # Check the verdict
                if "VERDICT: FALSE" in fact_check_output or "verdict: false" in fact_check_output.lower():
                    verdict_passed = False
                    print("\n" + "="*60)
                    print("❌ FACT CHECK FAILED - Research deemed inaccurate")
                    print("="*60 + "\n")
                    
                    # Extract the reason if available
                    if "REASON:" in fact_check_output:
                        reason_start = fact_check_output.find("REASON:")
                        reason_end = reason_start + 200
                        reason_text = fact_check_output[reason_start:reason_end]
                        print(f"Fact checker's feedback: {reason_text}\n")
                    
                elif "VERDICT: TRUE" in fact_check_output or "verdict: true" in fact_check_output.lower():
                    verdict_passed = True
                    print("\n" + "="*60)
                    print("✅ FACT CHECK PASSED - Research verified as accurate")
                    print("="*60 + "\n")
            else:
                # Fallback: check if verdict is in the final output
                result_str = str(result)
                if "VERDICT: FALSE" in result_str or "verdict: false" in result_str.lower():
                    verdict_passed = False
                    # Extract writer's response (everything after the last occurrence of REASON:)
                    if "REASON:" in result_str:
                        parts = result_str.split("REASON:")
                        if len(parts) > 1:
                            # Get text after reason
                            writer_output = parts[-1].split("\n", 1)[-1].strip()
                    
                    print("\n" + "="*60)
                    print("❌ FACT CHECK FAILED - Research deemed inaccurate")
                    print("="*60 + "\n")
            
            # If fact check passed, return the writer's answer
            if verdict_passed:
                # Clean up the writer's output - remove any verdict text if present
                clean_output = writer_output
                if "VERDICT:" in clean_output:
                    # Remove everything from VERDICT onwards
                    clean_output = clean_output.split("VERDICT:")[0].strip()
                
                return clean_output
            
            # If fact check failed and we have retries left
            if retries < max_retries:
                # Store the research output for retry context
                if tasks_output and len(tasks_output) >= 1:
                    research_output = str(tasks_output[0].raw) if hasattr(tasks_output[0], 'raw') else str(tasks_output[0])
                    previous_response = f"Research findings: {research_output}\n\nFact check result: {fact_check_output}"
                else:
                    previous_response = str(result)
                    
                retries += 1
                print(f"🔄 Retrying... (Attempt {retries + 1} of {max_retries + 1})\n")
                continue
            else:
                # Exhausted all retries
                print("\n" + "="*60)
                print("⚠️  Maximum retries reached. Returning best available answer.")
                print("="*60 + "\n")
                return (
                    f"Note: After {max_retries + 1} attempts, the system could not fully verify "
                    f"the answer. Here is the most recent response:\n\n"
                    f"{writer_output}"
                )
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return f"An error occurred: {str(e)}"

# Gradio Interface
def gradio_interface(query):
    if not query or not query.strip():
        return "Please enter a question."
    return run_rag_pipeline(query.strip())

# Create Gradio App
def create_gradio_app():
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(
            label="Ask a Question",
            placeholder="e.g., What is self-attention? or What are the latest AI developments?",
            lines=2
        ),
        outputs=gr.Textbox(label="Answer", lines=10),
        title="🤖 Agentic RAG: AI Knowledge Assistant",
        description="Ask questions about transformers/attention (searches PDF) or any topic (searches web). Powered by Google Gemini.",
        examples=[
            ["What is self-attention?"],
            ["Explain the transformer architecture"],
            ["How does multi-head attention work?"],
            ["What are the latest developments in AI?"],
        ],
    )
    return iface

# Launch the Gradio App
if __name__ == "__main__":
    print("Starting Agentic RAG system...")
    print("Using Google Gemini LLM and HuggingFace embeddings")
    app = create_gradio_app()
    app.launch(theme="soft")
