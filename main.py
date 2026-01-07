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
def create_agents():
    researcher = Agent(
        role="Research Specialist",
        goal="Find accurate and relevant information to answer questions",
        backstory=(
            "You are an expert researcher with access to two tools:\n"
            "1. PDF Search Tool: Use for questions about transformers, attention mechanisms, "
            "encoders, decoders, or the 'Attention is All You Need' paper\n"
            "2. Web Search Tool: Use for general questions, recent developments, or other topics\n"
            "Choose the most appropriate tool based on the question."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    writer = Agent(
        role="Answer Writer",
        goal="Create clear, accurate, and helpful answers",
        backstory=(
            "You are an expert at synthesizing information into clear answers. "
            "You always base your responses on the provided research and cite sources when relevant."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    return [researcher, writer]

# Task Definitions
def create_tasks(agents, tools):
    pdf_tool, web_tool = tools
    researcher, writer = agents

    research_task = Task(
        description=(
            "Research the following question: {question}\n\n"
            "Guidelines:\n"
            "- For questions about transformers, attention, encoders, or decoders: use PDF Search Tool\n"
            "- For other questions or recent information: use Web Search Tool\n"
            "- Gather comprehensive information to answer the question thoroughly"
        ),
        expected_output="Detailed research findings relevant to the question",
        agent=researcher,
        tools=[pdf_tool, web_tool],
    )

    writing_task = Task(
        description=(
            "Using the research provided, write a clear and accurate answer to: {question}\n\n"
            "Guidelines:\n"
            "- Be concise but comprehensive\n"
            "- Base your answer entirely on the research provided\n"
            "- Explain technical concepts clearly\n"
            "- If the information comes from the PDF, mention it's from the 'Attention is All You Need' paper"
        ),
        expected_output="A clear, well-written answer to the question",
        agent=writer,
        context=[research_task],
    )

    return [research_task, writing_task]

# Main RAG Function
def run_rag_pipeline(question):
    try:
        # Download PDF if not exists
        download_pdf()
        
        # Initialize vectorstore
        initialize_vectorstore()

        # Setup tools
        tools = [pdf_search_tool, web_search_tool]

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
        result = rag_crew.kickoff(inputs={"question": question})
        return str(result)
        
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
