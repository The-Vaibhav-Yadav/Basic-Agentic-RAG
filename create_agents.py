from crewai import Agent, Task
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get('GOOGLE_API_KEY'),
)

def create_agents():
    researcher = Agent(
        role="Research Specialist",
        goal="Find accurate and relevant information to answer questions",
        backstory=(
            "You are an expert researcher with access to two tools:\n"
            "1. PDF Search Tool: Use for questions about transformers, attention mechanisms, "
            "encoders, decoders, or the 'Attention is All You Need' paper\n"
            "2. Web Search Tool: Use for general questions, recent developments, or other topics\n\n"
            "Choose the most appropriate tool based on the question. "
            "If you see that a previous answer was marked as incorrect, carefully analyze "
            "what went wrong and try a different search approach or use different keywords."
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

    fact_checker = Agent(
        role="Fact Checker",
        goal="Check the accuracy of the information provided",
        backstory=(
            "You are an expert at checking the accuracy of information. You will be given a question and a list of sources. You will need to check the accuracy of the information provided and return a boolean value."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    return [researcher, writer, fact_checker]

# Task Definitions
def create_tasks(agents, tools):
    pdf_search_tool, web_search_tool = tools
    researcher, writer, fact_checker = agents

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
        tools=[pdf_search_tool, web_search_tool],
    )

    fact_checking_task = Task(
        description=(
            "Verify the factual accuracy of the research findings for the question: {question}\n\n"
            "Review the research sources carefully and verify the information is accurate.\n\n"
            "Guidelines:\n"
            "- Check if the research findings are factually accurate and reliable\n"
            "- Check if the research actually addresses the question\n"
            "- Look for any contradictions or questionable information in the research\n"
            "- Verify the sources are appropriate (PDF for transformer questions, web for others)\n\n"
            "You MUST respond in this exact format:\n"
            "VERDICT: TRUE (if research is accurate and reliable) or VERDICT: FALSE (if research is questionable)\n"
            "REASON: [Brief explanation of your verdict]"
        ),
        expected_output="A verdict (TRUE/FALSE) with a brief reason",
        agent=fact_checker,
        context=[research_task],
    )

    writing_task = Task(
        description=(
            "The research has been verified as accurate. Now write a clear and comprehensive answer to: {question}\n\n"
            "Guidelines:\n"
            "- Be concise but comprehensive\n"
            "- Base your answer entirely on the verified research provided\n"
            "- Explain technical concepts clearly\n"
            "- If the information comes from the PDF, mention it's from the 'Attention is All You Need' paper\n"
            "- Do NOT include the fact checker's verdict in your response\n"
            "- Write only the answer that will be shown to the user"
        ),
        expected_output="A clear, well-written answer to the question (only the answer, no verdict)",
        agent=writer,
        context=[research_task, fact_checking_task],
    )

    return [research_task, fact_checking_task, writing_task]