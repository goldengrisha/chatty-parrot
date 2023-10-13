## Reply.io ChatBot

### Description

This is a ChatBot powered by the OpenAI GPT-3.5 and GPT-4 models, and other libraries for document loading, text splitting, and more.   
It can answer questions based on provided PDF files or URLs.

### Features

- PDF Document Loading: Load text from PDF documents using PyPDFLoader.
- Web Content Retrieval: Retrieve content from web pages using WebBaseLoader or Selenium(Image Recognition).
- Text Splitting: Split long documents into manageable chunks using RecursiveCharacterTextSplitter.
- Natural Language Processing: Utilize Hugging Face embeddings for text processing.
- Contextual Chatbot: Engage in contextual conversations with the user.
- Memory Functionality: Keep track of chat history and context.
- Internet Access: Retrieve information from the web using Google Search API.
- Question Answering: Answer questions based on documents and context.
- Friendly Customer Support Agent: Provides helpful and factual information to users.

### Installation

1. Clone the repository to your local machine:  
`git clone https://github.com/goldengrisha/chatty-parrot/tree/main`

2. Open a terminal in the project folder
3. Install the required dependencies using Poetry:  
`poetry install`

## Usage
### Initialization
To initialize the ChatBot, you can configure its behavior by editing the .env file:


TELEGRAM_BOT_TOKEN= set an authentication token for a Telegram Bot  
OPENAI_API_KEY= set an API key for the GPT-4 model  
CHUNK_SIZE= set the size of text chunks used for processing documents or text data   
CHUNK_OVERLAP= set how much overlap there should be between consecutive text chunks  
SEARCH_PRECISION=set to control the precision or accuracy of a search operation  
GOOGLE_CSE_ID= set the Custom Search Engine (CSE) ID for Google Custom Search  
GOOGLE_API_KEY= set an API key for accessing Google services  
 
Once configured, run the ChatBot:
`poetry run python chatty_perrot.py`

The ChatBot will initialize based on your configuration, and you can engage in conversations with it.

So you can go in telegram and start chatting with bot.
