import aiofiles
import logging
import os

from typing import Any, Dict

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from urllib.parse import urlparse

from modules.sales_chat_bot import RetrievalChatBot

from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from modules.settings import Settings

form_router = Router()
bot = Bot(token=Settings.get_tg_token(), parse_mode=ParseMode.HTML)
dp = Dispatcher()
chat_bot = RetrievalChatBot()


class Processor(StatesGroup):
    chat_bot_type = State()
    is_with_memory = State()
    is_with_context = State()
    is_with_internet_access = State()
    regular_usage = State()
    change_context = State()
    waiting_for_file = State()
    waiting_for_url = State()
    change_url_process = State()
    processing_url = State()

    salesperson_name = State()
    salesperson_role = State()
    company_name = State()
    company_business = State()
    company_values = State()
    conversation_purpose = State()
    conversation_type = State()
    conversation_stage = State()

    async def run(self):
        global bot
        global dp
        dp.include_router(form_router)
        await dp.start_polling(bot)


@form_router.message(CommandStart())
async def command_start(message: Message, state: FSMContext) -> None:
    await state.set_state(Processor.salesperson_name)
    await message.answer(
        "Hi, let's configure the bot👋\n<b>Please enter the bot's name: </b>(e.g. Ted Lasso)",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.salesperson_name)
async def process_salesperson_name(message: Message, state: FSMContext) -> None:
    salesperson_name = message.text
    await state.update_data(salesperson_name=salesperson_name)
    await state.set_state(Processor.salesperson_role)
    await message.answer(
        "<b>Please enter the bot's role: </b>(e.g. Business Development Representative)",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.company_name)
async def process_company_name(message: Message, state: FSMContext) -> None:
    company_name = message.text
    await state.update_data(company_name=company_name)
    await state.set_state(Processor.company_business)
    await message.answer(
        "<b>Please enter the company's business description: </b>"
        "\n(e.g.  Reply is your AI-powered sales engagement platform to create new opportunities at scale – automatically.)",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.salesperson_role)
async def process_salesperson_role(message: Message, state: FSMContext) -> None:
    salesperson_role = message.text
    await state.update_data(salesperson_role=salesperson_role)
    await state.set_state(Processor.company_name)
    await message.answer(
        "<b>Please enter the company name: </b>(e.g. Reply)",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.company_business)
async def process_company_business(message: Message, state: FSMContext) -> None:
    company_business = message.text
    await state.update_data(company_business=company_business)
    await state.set_state(Processor.company_values)
    await message.answer(
        "<b>Please enter the company's core values and mission: </b>"
        "\n (e.g. Our mission is to connect businesses through personalized communication at scale.)",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.company_values)
async def process_company_values(message: Message, state: FSMContext) -> None:
    company_values = message.text
    await state.update_data(company_values=company_values)
    await state.set_state(Processor.conversation_purpose)
    await message.answer(
        "<b>Please describe the purpose of this conversation: </b>"
        "\n (e.g. Help to find information what they are looking for.)",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.conversation_purpose)
async def process_conversation_purpose(message: Message, state: FSMContext) -> None:
    conversation_purpose = message.text
    await state.update_data(conversation_purpose=conversation_purpose)
    await state.set_state(Processor.conversation_type)
    await message.answer(
        "<b>Please enter the conversation type: </b>(e.g., call, email, meeting)",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.conversation_type)
async def process_conversation_type(message: Message, state: FSMContext) -> None:
    conversation_type = message.text
    data = await state.update_data(conversation_type=conversation_type)
    await state.set_state(Processor.regular_usage)

    await show_summary(message=message, data=data, keyboard=ReplyKeyboardRemove())
    keyboard = await create_regular_usage_keyboard()

    await message.answer(
        "Thank you! The bot has been configured.",
        reply_markup=keyboard,
    )


# @form_router.message(CommandStart())
# async def command_start(message: Message, state: FSMContext) -> None:
#     """
#     Handle the /start command to initiate the chatbot configuration.
#     """
#     await state.set_state(Processor.chat_bot_type)
#     await message.answer(
#         f"Hi, I am a ContextGPT bot👋 \nPlease select chatbot type:",
#         reply_markup=ReplyKeyboardMarkup(
#             keyboard=[
#                 [
#                     KeyboardButton(text="GPT-3.5-Turbo"),
#                     KeyboardButton(text="GPT-4"),
#                 ]
#             ],
#             resize_keyboard=True,
#         ),
#     )


@form_router.message(Processor.chat_bot_type, F.text.in_({"GPT-3.5-Turbo", "GPT-4"}))
async def process_chat_bot_type(message: Message, state: FSMContext) -> None:
    """
    Handle the user's selection of the chatbot type.
    """
    await state.update_data(chat_bot_type=message.text)
    await state.set_state(Processor.is_with_memory)
    keyboard = await create_yes_no_keyboard()
    await message.answer(
        f"Would you like to use memory?",
        reply_markup=keyboard,
    )


@form_router.message(Processor.is_with_memory, F.text.in_({"Yes", "No"}))
async def process_memory(message: Message, state: FSMContext) -> None:
    """
    Handle the user's choice to use memory.
    """
    await state.update_data(is_with_memory=message.text)
    await state.set_state(Processor.is_with_context)
    keyboard = await create_yes_no_keyboard()
    await message.answer(
        f"Would you like to use context or full chat gpt?",
        reply_markup=keyboard,
    )


@form_router.message(Processor.is_with_context, F.text.in_({"Yes", "No"}))
async def process_context(message: Message, state: FSMContext) -> None:
    """
    Handle the user's choice to use context.
    """
    await state.update_data(is_with_context=message.text)
    await state.set_state(Processor.is_with_internet_access)
    keyboard = await create_yes_no_keyboard()
    await message.answer(
        f"Would you like to use internet access?",
        reply_markup=keyboard,
    )


@form_router.message(
    Processor.is_with_internet_access,
    F.text.in_({"Yes", "No"}),
)
async def process_internet_access(message: Message, state: FSMContext) -> None:
    """
    Handle the user's choice to use internet access.
    """
    data = await state.update_data(is_with_internet_access=message.text)
    await state.set_state(Processor.regular_usage)

    await show_summary(message=message, data=data, keyboard=ReplyKeyboardRemove())
    keyboard = await create_regular_usage_keyboard()
    await message.answer(
        "You can use your bot now. Please select the options below⬇️",
        reply_markup=keyboard,
    )


@form_router.message(Processor.regular_usage, F.text.in_({"Show status"}))
async def process_regular_usage_show_status(
    message: Message, state: FSMContext
) -> None:
    """
    Handle the command to show the current bot configuration status.
    """
    keyboard = await create_regular_usage_keyboard()
    await show_summary(
        message=message,
        data=await state.get_data(),
        keyboard=keyboard,
    )


@form_router.message(Processor.regular_usage, F.text.in_({"Reset", "/reset"}))
async def process_regular_usage_reset(message: Message, state: FSMContext) -> None:
    """
    Handle the command to reset the bot's configuration and state.
    """
    chat_bot.initialized = False
    await state.set_data({})
    await state.set_state(Processor.salesperson_name)
    await message.answer(
        "Let's configure your bot again.\nPlease enter the salesperson's name:"
    )


@form_router.message(Processor.regular_usage, F.text.casefold() == "/uploadpdf")
async def process_uploadPDF(message: Message, state: FSMContext) -> None:
    """
    Handle the command to upload a PDF file.
    """
    await state.set_state(Processor.waiting_for_file)
    await message.answer("Please, upload PDF file.", reply_markup=ReplyKeyboardRemove())


@form_router.message(Processor.regular_usage, F.text.in_({"Upload PDF file or URL"}))
async def process_regular_usage_document(message: Message, state: FSMContext) -> None:
    """
    Handle the command to choose whether to upload a PDF file or URL.
    """
    await state.set_state(Processor.change_context)
    await message.answer(
        f"Would you like upload PDF file or url?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="PDF File"),
                    KeyboardButton(text="Url"),
                ]
            ]
        ),
    )


@form_router.message(Processor.change_context, F.text.in_({"PDF File"}))
async def process_change_context_document(message: Message, state: FSMContext) -> None:
    """
    Handle the choice to upload a PDF file.
    """
    await state.set_state(Processor.waiting_for_file)
    await message.answer(
        "Please, upload a pdf file.", reply_markup=ReplyKeyboardRemove()
    )


@form_router.message(Processor.change_context, F.text.in_({"Url"}))
async def process_change_context_document(message: Message, state: FSMContext) -> None:
    """
    Handle the choice to upload a URL.
    """
    await state.update_data(path=message.text, is_file=False)
    await state.set_state(Processor.change_url_process)
    keyboard = await create_url_process_keyboard()
    await message.answer(
        "What process do you want to perform with the URL?",
        reply_markup=keyboard,
    )


@form_router.message(Processor.regular_usage, F.text.casefold() == "/sendurl")
async def process_send_url(message: Message, state: FSMContext) -> None:
    """
    Handle the command to send a URL.
    """
    await state.set_state(Processor.change_url_process)
    keyboard = await create_url_process_keyboard()
    await message.answer(
        "What process do you want to perform with the URL?",
        reply_markup=keyboard,
    )


@form_router.message(
    Processor.change_url_process, F.text.in_({"Loader", "Image Recognition"})
)
async def process_change_url_process(message: Message, state: FSMContext) -> None:
    """
    Handle the choice of what process to perform with the URL.
    """
    url_process = message.text
    await state.update_data(url_process=url_process)
    await state.set_state(Processor.processing_url)
    await message.answer(
        f"You have selected {url_process}. Please, paste the URL here.",
        reply_markup=ReplyKeyboardRemove(),
    )


@form_router.message(Processor.processing_url)
async def process_processing_url(message: Message, state: FSMContext) -> None:
    """
    Handle the URL processing, e.g., web page loading and image recognition.
    """
    url = message.text

    # Check if it's a valid URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        await message.answer("Please, paste a valid URL.")
        return
    url_process = (await state.get_data())["url_process"]
    await state.update_data(path=message.text, is_file=False, url_process=url_process)
    await state.set_state(Processor.regular_usage)
    keyboard = await create_regular_usage_keyboard()
    await message.answer(
        f"Your url has been uploaded. What is your question?",
        reply_markup=keyboard,
    )


@form_router.message(Processor.waiting_for_url)
def process_invalid_url(message: Message, state: FSMContext) -> None:
    """
    Handle the case where the user enters an invalid URL.
    """
    message.answer("Invalid url. Please paste here valid URL.")


@form_router.message(Processor.waiting_for_file, F.document)
async def process_waiting_for_file(message: Message, state: FSMContext) -> None:
    """
    Handle the user uploading a document.
    """
    try:
        if not os.path.exists("downloads"):
            os.makedirs("downloads")
        # Download the document
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path
        if not file_path.endswith(".pdf"):
            await message.answer("Please upload a PDF file.")
        download_file = await bot.download_file(file_path)
        file_name = message.document.file_name
        local_path = f"downloads/{file_name}"
        print(local_path)

        async with aiofiles.open(local_path, mode="wb") as local_file:
            await local_file.write(download_file.read())

        await state.update_data(path=local_path, is_file=True)
        keyboard = await create_regular_usage_keyboard()
        await message.answer(
            f"Your file {file_name} has been uploaded. What is your question?",
            reply_markup=keyboard,
        )
        await state.set_state(Processor.regular_usage)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


@form_router.message(Processor.waiting_for_file)
def process_invalid_file(message: Message, state: FSMContext) -> None:
    """
    Handle the case where the user uploads an invalid file.
    """
    message.answer("Please upload a file.")


@form_router.message(Processor.regular_usage, F.text.casefold() == "/help")
async def process_regular_usage_reset(message: Message, state: FSMContext) -> None:
    """
    Handle the command to display the help message.
    """
    await print_help(message)


@form_router.message(Processor.regular_usage)
async def process_regular_usage_reset(message: Message, state: FSMContext) -> None:
    """
    Handle regular usage of the bot, processing user queries.
    """
    global chat_bot
    data = await state.get_data()
    print("regular usage", data.get("path"))
    keyboard = await create_regular_usage_keyboard()
    if not all([data.get("path")]):
        await message.answer(
            "Please, upload PDF file or url first.",
            reply_markup=keyboard,
        )
        return

    data["use_tools"] = True
    logging.info(f"DATA: {data}")

    config = dict(
        salesperson_name=data.get("salesperson_name", "John"),
        salesperson_role=data.get(
            "salesperson_role", "Business Development Representative"
        ),
        company_name=data.get("company_name", "Reply.io"),
        company_business=data.get(
            "company_business",
            "We are your AI-powered sales engagement platform to create new opportunities at scale – automatically.",
        ),
        company_values=data.get(
            "company_values",
            "Our mission is to connect businesses through personalized communication at scale.",
        ),
        conversation_purpose=data.get(
            "conversation_purpose",
            "Help to find information what they are looking for.",
        ),
        conversation_history=[],
        conversation_type=data.get("conversation_type", "call"),
        conversation_stage=(
            "Introduction: Start the conversation by introducing yourself and your company.",
            "Be polite and respectful while keeping the tone of the conversation professional.",
        ),
    )

    if not chat_bot.initialized:
        chat_bot.initialize(
            data.get("use_tools"),
            "gpt-3.5-turbo",
            data.get("is_file"),
            data.get("path"),
            config=config
            # data.get("url_process"),
        )

    if chat_bot.path != data.get("path") or chat_bot.is_file != data.get("is_file"):
        chat_bot.initialize(
            data.get("use_tools"),
            "gpt-3.5-turbo",
            data.get("is_file"),
            data.get("path"),
            # data.get("url_process"),
        )
    # answer = chat_bot.query_executor.invoke(message.text)

    # self.query_executor.human_step("Can you compare open and reply rates?")
    #         self.query_executor.step()

    chat_bot.query_executor.human_step(message.text)
    chat_bot.query_executor.step()
    output = (
        chat_bot.query_executor.conversation_history[-1]
        .replace("<END_OF_TURN>", "")
        .strip()
    )

    # logging.error(f"\n\n\n{type(output)}\n\n\n")

    # output = (
    #     answer.get("output")
    #     if answer.get("output")
    #     else answer.get("result", "No output available")
    # )
    await message.reply(output)


@form_router.message(Processor.chat_bot_type)
async def process_unknown_write_bots(message: Message) -> None:
    """
    Handle the scenario where the user didn't select a chatbot type.
    """
    await message.reply("Choose the openai model firstly:")


@form_router.message(Processor.is_with_memory)
async def process_unknown_write_bots(message: Message) -> None:
    """
    Handle the scenario where the user didn't select whether to use memory.
    """
    await message.reply("You need to choose whether to use memory in the bot:")


@form_router.message(Processor.is_with_context)
async def process_unknown_write_bots(message: Message) -> None:
    """
    Handle the scenario where the user didn't select whether to use context.
    """
    await message.reply("Choose using context instead of full chat gpt:")


@form_router.message(Processor.is_with_internet_access)
async def process_unknown_write_bots(message: Message) -> None:
    """
    Handle the scenario where the user didn't select whether to use internet access.
    """
    await message.reply("Do you want to use access to the internet?")


async def show_summary(
    message: Message,
    data: Dict[str, Any],
    keyboard,
    positive: bool = True,
) -> None:
    """
    Show a summary of the bot's configuration to the user.
    """
    salesperson_name = data.get("salesperson_name", "Ted Lasso")
    salesperson_role = data.get(
        "salesperson_role", "Business Development Representative"
    )
    company_name = data.get("company_name", "Reply")
    company_business = data.get(
        "company_business",
        "Reply is your AI-powered sales engagement platform to create new opportunities at scale – automatically.",
    )
    company_values = data.get(
        "company_values",
        "Our mission is to connect businesses through personalized communication at scale.",
    )
    conversation_purpose = data.get(
        "conversation_purpose", "Help to find information what they are looking for."
    )
    conversation_type = data.get("conversation_type", "call")

    text = f"""
        Hey, your bot has been created with the following settings: 
        Chatbot name: {salesperson_name} 
        Chatbot role: {salesperson_role} 
        Company name: {company_name}
        Company business: {company_business}
        Company values: {company_values}
        Conversation purpose: {conversation_purpose}
        Conversation type: {conversation_type}"""

    await message.answer(text=text, reply_markup=keyboard)


@form_router.message(F.text.casefold() == "/help")
async def command_help(message: Message, state: FSMContext):
    """
    Handle the user's request for help by displaying the help message.
    """
    help_message = """
    Here are the available commands:
    /start - start the bot
    /reset - reset the context
    /uploadPDF - upload new PDF file
    /sendURL - send new URL

    /help - show this help message
    """
    await print_help(message)


async def print_help(message: Message):
    """
    Display the help message to the user.
    """
    help_message = """
    Here are the available commands:
    /start - start the bot
    /reset - reset the context
    /uploadPDF - upload new PDF file
    /sendURL - send new URL

    /help - show this help message
    """
    await message.answer(help_message, parse_mode=ParseMode.MARKDOWN)


async def create_url_process_keyboard():
    """
    Create a custom keyboard for selecting URL processing options.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="Loader"),
                KeyboardButton(text="Image Recognition"),
            ],
        ],
        resize_keyboard=True,
    )


async def create_yes_no_keyboard():
    """
    Create a custom keyboard for selecting "Yes" or "No" options.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="Yes"),
                KeyboardButton(text="No"),
            ],
        ],
        resize_keyboard=True,
    )


async def create_regular_usage_keyboard():
    """
    Create a custom keyboard for regular usage options.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="Upload PDF file or URL"),
            ],
            [
                KeyboardButton(text="Reset"),
                KeyboardButton(text="Show status"),
            ],
        ],
        resize_keyboard=True,
    )
