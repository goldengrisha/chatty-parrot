# from langchain.chat_models import ChatOpenAI
# from dotenv import load_dotenv
#
# from modules.sales_chat_bot import SalesGPT, conversation_stages
#
# load_dotenv()
#
#
# class ChatBot:
#     def __init__(self) -> None:
#         self.initialized = False
#
#     def initialize(
#         self,
#         # chat_bot_type: str,
#         # is_with_memory: bool,
#         # is_with_context: bool,
#         # is_with_internet_access: bool,
#         use_tools: bool,
#         is_file: bool,
#         path: str,
#         url_process: str,
#     ) -> None:
#         """
#         Initialize the ChatBot class with settings and models.
#
#         Args:
#             chat_bot_type (str): Type of chatbot.
#             is_with_memory (bool): Whether to include memory.
#             is_with_context (bool): Whether to include context.
#             is_with_internet_access (bool): Internet access.
#             is_file (bool): Whether to process a file (PDF or URL).
#             path (str): Path to the file or URL.
#             url_process (str): URL processing method ("Loader" or "Screenshot").
#         """
#         self.use_tools = use_tools
#         self.path = path
#         self.is_file = is_file
#         self.url_process = url_process
#         # model = ChatOpenAI(temperature=0, model_name=chat_bot_type)
#         # embeddings = self.get_embeddings()
#         # if is_file:
#         #     documents = self.load_pdf(path)
#         # else:
#         #     if url_process == "Loader":
#         #         documents = self.load_url(path)
#         #     else:
#         #         documents = self.read_url_webdriver_screenshot(path)
#         # chunked_documents = self.split_documents(documents)
#         # chat_bot: Any = self.get_chat_bot(
#         #     chunked_documents, embeddings, model, is_with_memory, is_with_context
#         # )
#         #
#         # if is_with_internet_access:
#         #     self.query_executor = self.get_agent(chat_bot, model, is_with_memory)
#         # else:
#         #     self.query_executor = chat_bot
#         # self.initialized = True
#         config = dict(
#             salesperson_name="Ted Lasso",
#             salesperson_role="Business Development Representative",
#             company_name="Reply",
#             company_business="Reply is your AI-powered sales engagement platform to create new opportunities at scale – automatically.",
#             company_values="Our mission is to connect businesses through personalized communication at scale.",
#             conversation_purpose="Help to find information what they are looking for.",
#             conversation_history=[],
#             conversation_type="call",
#             conversation_stage=conversation_stages.get(
#                 "1",
#                 "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional.",
#             ),
#             use_tools=True,
#             is_file=is_file,
#             path=path,
#             url_process=url_process,
#             # path="How_Reply_Generated_400k_Case_Study.pdf",
#         )
#         verbose = True
#         llm = ChatOpenAI(temperature=0.9)
#         sales_agent = SalesGPT.from_llm(
#             llm,
#             verbose=verbose,
#             **config,
#         )
#         # init sales agent
#         sales_agent.seed_agent()
#         sales_agent.step()
#
#         sales_agent.human_step("Can you compare open and reply rates?")
#         sales_agent.step()
