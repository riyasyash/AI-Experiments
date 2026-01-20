# NOTE:
# This project intentionally uses message-based state via AutoGen.
# For production systems, this would be replaced with an explicit
# shared state object owned by the orchestrator.

from dotenv import load_dotenv 
import os
# from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from typing import List, Sequence
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
import requests
import xml.etree.ElementTree as ET
import asyncio

load_dotenv()
# model_cli = OpenAIChatCompletionClient(model="gpt-4o",api_key=os.getenv("OPENAI_API_KEY"))
model_cli = OllamaChatCompletionClient(model="llama3",host="http://localhost:11434")

def arxiv_search_tool(query:str, max_results: int=3):
    base_url = f'http://export.arxiv.org/api/query'
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        root=ET.fromstring(resp.text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
        }
        entries = root.findall("atom:entry", ns)
        if not entries:
            return f"No papers found for topic '{query}'."
        entries_list = [f"Papers for '{query}':"]
        for i,entry in enumerate(entries, start=1):
            title = entry.find("atom:title", ns).text.strip()
            link = entry.find("atom:id", ns).text.strip()
            summary = entry.find("atom:summary", ns).text.replace("\n", " ")
            entries_list.append(f"{i}. {title}\n   Link: {link}\n   Abstract: {summary[:300]}...")

        return "\n".join(entries_list)
    except Exception as e:
        print(e)

topic_agent = AssistantAgent(
    name = "topic_agent",
    model_client=model_cli,
    description = "Refines Broad Research ideas",
    system_message = """
    ROLE: You are a research topic refiner agent.
    TASK: Given a general idea, output one refined research topic, 
    
    OUTPUT FORMAT: 
    Refined Research Topic: <Topic> 
    End with: TASK_COMPLETED_TOPIC_AGENT 
    """
)

# topic_result = await topic_agent.run(task= "AI in healthcare")
# print(topic_result.messages[-1].content)

paper_agent = AssistantAgent(
    name = "paper_agent",
    tools = [arxiv_search_tool],
    model_client = model_cli,
    description = "Finds research papers",
    system_message = """
    ROLE: You are a reserach paper finder
    TASK: Find 2-3 recent research papers relevant to the topic
    OUTPUT FORMAT:
    ### Topic: <Topic>
    - Paper 1: <Title> - <Summary> - <URL>
    - Paper 2: <Title> - <Summary> - <URL>
    
    """,
)

# paper_result = await paper_agent.run(task=topic_result.messages[-1].content)
# print(paper_result.messages[-1].cotent)

insight_agent = AssistantAgent(
    name="insight_agent",
    model_client=model_cli,
    description="Summarise key ideas from discovered research papers",
    system_message="""
    ROLE: You are a research paper summarising agent
    TASK: Summarise the given research papers and create 3-5 concise bullet points summarizing the common findings

    OUTPUT FORMAT:
    Key Insights
    - <Insight 1>
    - <Insight 2>
    """

)

# insight_result = await insight_agent.run(task=paper_result.messages[-1].content)
# print(insight_result.messages[-1].cotent)

report_agent = AssistantAgent(
    name="report_agent",
    model_client=model_cli,
    description="Compile Concise reseacrh report",
    system_message="""
    ROLE: You are a research insights reporting agent
    TASK: Compile the given research insights to a structured report

    OUTPUT FORMAT:
    ### Research Report
    ## Introduction
    ...
    ## Key Findigs
    ...
    ## Discussion
    ...
    ## Conclusion
    ...
    """

)

# report_result = await report_agent.run(task=insight_result.messages[-1].content)
# print(report_result.messages[-1].cotent)

gap_agent = AssistantAgent(
    name="gap_agent",
    model_client=model_cli,
    description="Perform Research Gap Analysis",
    system_message="""
    ROLE: You are a research gap analyser
    TASK: Analyse the research report and comeup with 2-3 Research Gaps and 2-3 Proposed Next Steps

    OUTPUT FORMAT:
    ### Research Gaps
    <Gap 1>
    <Gap 2>
    ## Proposed Next Steps
    <Step 1>
    <Step 2>

    FINAL REPORT
    """

)
# gap_result = await gap_agent.run(task=report_result.messages[-1].content)
# print(gap_result.messages[-1].cotent)

text_mention_termination = TextMentionTermination("FINAL REPORT")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = max_messages_termination | text_mention_termination



user_proxy_agent = UserProxyAgent(
    "UserProxyAgent",
    description = "A proxy for the user to approve or disapprove tasks"
)

user_approved = False
def selector_func_with_user_proxy(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str|None:
    global user_approved
    if len(messages) == 0:
        return topic_agent.name
    last_source = messages[-1].source
    if last_source == topic_agent.name and not user_approved:
        return user_proxy_agent.name
    if last_source == user_proxy_agent.name:
        content = messages[-1].content.upper()
        if "APPROVE" in content:
            user_approved = True
            return paper_agent.name
        else:
            return topic_agent.name
    if user_approved:
        return None
    return None

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the topic agent has assigned tasks before other agents start working.
Only select one agent.
"""

async def main():
    task = "Ayurveda for stroke patients"

    # team = SelectorGroupChat(
    #     [topic_agent, paper_agent, insight_agent, report_agent, gap_agent, user_proxy_agent],
    #     model_client=model_cli,
    #     termination_condition=termination,
    #     selector_prompt=selector_prompt,
    #     selector_func=selector_func_with_user_proxy,
    #     allow_repeated_speaker=False
    # )
    team = RoundRobinGroupChat(
        [topic_agent, paper_agent, insight_agent, report_agent, gap_agent],
        termination_condition=termination
    )

    await Console(team.run_stream(task=task))

if __name__ == "__main__":
    asyncio.run(main())