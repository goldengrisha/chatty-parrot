import os
import langchain
import logging
import requests

from typing import List, Any, Optional, Sequence

from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch, Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents.types import AgentType
from langchain.agents import ZeroShotAgent, AgentExecutor, initialize_agent
from langchain.chains import LLMChain
from PIL import Image
from selenium import webdriver
import pytesseract

from dotenv import load_dotenv

load_dotenv()


class ChatBot:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(
        self,
        chat_bot_type: str,
        is_with_memory: bool,
        is_with_context: bool,
        is_with_internet_access: bool,
        is_file: bool,
        path: str,
        url_process: str,
    ):
        self.path = path
        self.is_file = is_file
        self.url_process = url_process
        model = ChatOpenAI(temperature=0, model_name=chat_bot_type)
        embeddings = self.get_embeddings()
        if is_file:
            documents = self.load_pdf(path)
        else:
            if url_process == "Loader":
                documents = self.load_url(path)
            else:
                documents = self.read_url_webdriver_screenshot(path)
        chunked_documents = self.split_documents(documents)
        chat_bot: Any = self.get_chat_bot(
            chunked_documents, embeddings, model, is_with_memory, is_with_context
        )

        if is_with_internet_access:
            self.query_executor = self.get_agent(chat_bot, model, is_with_memory)
        else:
            self.query_executor = chat_bot
        self.initialized = True

    def load_pdf(self, pdf_path: str) -> List[Document]:
        loader = PyPDFLoader(pdf_path)
        raw_pages = loader.load_and_split()

        return raw_pages

    def load_url(self, url: str) -> List[Document] | None:
        try:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
            loader = WebBaseLoader([url])
            loader.requests_kwargs = {
                "verify": False,
                "headers": {
                    "User-Agent": user_agent,
                },
            }
            documents = loader.load()
            return documents

        except Exception as e:
            logging.error(f"An error occurred: {e}")

        return None

    def read_url_webdriver_screenshot(self, url: str) -> List[Document] | None:
        try:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"

            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"--user-agent={user_agent}")

            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)

            driver.execute_script("document.body.style.zoom='100%'")
            page_width = driver.execute_script(
                "return Math.max(document.body.scrollWidth, document.body.offsetWidth, document.documentElement.clientWidth, document.documentElement.scrollWidth, document.documentElement.offsetWidth);"
            )
            page_height = driver.execute_script(
                "return Math.max(document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);"
            )

            driver.set_window_size(page_width, page_height)

            screenshot_path = "full_page_screenshot.png"
            driver.save_screenshot(screenshot_path)

            img = Image.open(screenshot_path)
            text = pytesseract.image_to_string(img)
            doc = Document(page_content=text, metadata={"source": "local"})
            img.close()

            driver.quit()
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)

            return [doc]

        except requests.HTTPError as e:
            logging.error(f"Bad response from URL: {str(e)}")
        except requests.RequestException as e:
            logging.error(f"Couldn't retrieve text from URL: {str(e)}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        return None

    def split_documents(self, documents: List[Document]) -> Sequence[Document] | None:
        # Text splitter
        chunk_size = 1000
        chunk_overlap = 0
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunked_documents = text_splitter.transform_documents(documents=documents)
        if chunked_documents:
            return chunked_documents
        else:
            return None

    def get_embeddings(self) -> Any:
        model_name = "thenlper/gte-base"
        model_kwargs = {"device": "cpu"}
        hf_embedding = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs
        )

        return hf_embedding

    def get_chat_bot(
        self,
        chunked_documents: List[Document],
        embeddings: Any,
        model: ChatOpenAI,
        is_with_memory: bool,
        is_with_context: bool,
    ) -> RetrievalQA:
        db = Chroma.from_documents(chunked_documents, embeddings)
        retriever = db.as_retriever(
            search_kwargs={"score_threshold": 0.8}, kwargs={"top_k": 2}
        )
        prompt_messages = None
        if is_with_memory:
            prompt_messages = [
                SystemMessage(
                    content=(
                        "You are a sales person."
                        "You should help clients with their concerns about their choices and provide helpful solutions."
                        "You can ask questions to help you understand the problem."
                        "You should only talk within the context of problem."
                        if is_with_context
                        else ""
                        "Your main goal is getting contact info, then starting a trial."
                        "If you are unsure of how to help, you can suggest the client leave the contact like email or phone."
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    f"Answer the question based {'only' if is_with_context else ''} on the following context or history."
                ),
                HumanMessagePromptTemplate.from_template("History: {history}."),
                HumanMessagePromptTemplate.from_template("Context: {context}."),
                HumanMessagePromptTemplate.from_template("Question: {question}."),
            ]
        else:
            prompt_messages = [
                SystemMessage(
                    content=(
                        "You are a sales person."
                        "You should help clients with their concerns about their choices and provide helpful solutions."
                        "You can ask questions to help you understand the problem."
                        "You should only talk within the context of problem."
                        if is_with_context
                        else ""
                        "Your main goal is getting contact info, then starting a trial."
                        "If you are unsure of how to help, you can suggest the client leave the contact like email or phone."
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    f"Answer the question based {'only' if is_with_context else ''} on the following context."
                ),
                HumanMessagePromptTemplate.from_template("Context: {context}."),
                HumanMessagePromptTemplate.from_template("Question: {question}."),
            ]

        input_variables = (
            ["context", "question", "history"]
            if is_with_memory
            else ["context", "question"]
        )

        prompt = ChatPromptTemplate(
            messages=prompt_messages, input_variables=input_variables
        )

        chain_type_kwargs = (
            {
                "prompt": prompt,
                "memory": ConversationSummaryBufferMemory(
                    llm=model,
                    memory_key="history",
                    input_key="question",
                    max_token_limit=1000,
                ),
            }
            if is_with_memory
            else {"prompt": prompt}
        )

        retrieval_qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            verbose=True,
            chain_type_kwargs=chain_type_kwargs,
        )

        return retrieval_qa

    def get_agent(
        self,
        retrieval_qa: RetrievalQA,
        model: ChatOpenAI,
        is_with_memory: bool,
    ) -> AgentExecutor:
        search = GoogleSearchAPIWrapper()
        tools = [
            Tool(
                name="default",
                func=retrieval_qa.run,
                description="""Useful for when you need to answer specific questions""",
            ),
            Tool(
                name="search",
                func=search.run,
                description="Useful for when you need to search for specific information online",
            ),
        ]

        prefix = """
        You are a customer support agent. 
        You are designed to be as helpful as possible while providing only factual information. 
        You should be friendly, but not overly chatty.
        Your main goal is getting contact info, then starting a trial.
        If you are unsure of how to help, you can suggest the client leave the contact like email or phone.
        If you find answers in history just answer.
        You have access to the following tools:
        """
        suffix = ""
        promt = None
        memory = None
        if is_with_memory:
            suffix = """Begin!"
            {chat_history}
            Question: {input}
            {agent_scratchpad}
            """
            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )
            memory = ConversationBufferMemory(
                memory_key="chat_history", max_token_limit=500
            )

        else:
            suffix = """Begin!"
            Question: {input}
            {agent_scratchpad}
            """

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "agent_scratchpad"],
            )

        llm_chain = LLMChain(llm=model, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

        return agent


# chat_bot = ChatBot(
#     chat_bot_type="gpt-4",
#     is_with_memory=False,
#     is_with_context=True,
#     is_with_internet_access=True,
#     is_file=True,
#     path="./How_Reply_Generated_400k_Case_Study.pdf",
#     url_process="Loader"
# )

# print(chat_bot.query_executor.invoke("what is CEO of Google?"))
# print(chat_bot.query_executor.invoke("Can you compare open and reply rates?"))
# print(chat_bot.query_executor.invoke("What was the last question about?"))
