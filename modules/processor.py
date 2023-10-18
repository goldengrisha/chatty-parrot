import aiofiles
import logging
import os

from typing import Any, Dict

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from langchain.chat_models import ChatOpenAI
from urllib.parse import urlparse

from modules.sales_chat_bot import RetrievalChatBot, SalesGPT

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

user_bots: Dict[str, SalesGPT] = {}


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
    await message.answer("Hi, let's configure the bot! 👋")
    await message.answer(
        "<b>Please enter the bot's name or choose one below: </b>\n(1/7)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Gregory"),
                    KeyboardButton(text="Dmytro"),
                    KeyboardButton(text="Oksana"),
                    KeyboardButton(text="Mykyta"),
                ],
            ],
            resize_keyboard=True,
        ),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.salesperson_name)
async def process_salesperson_name(message: Message, state: FSMContext) -> None:
    salesperson_name = message.text
    await state.update_data(salesperson_name=salesperson_name)
    await state.set_state(Processor.salesperson_role)
    await message.answer(
        "<b>Please enter the bot's role or choose one below: </b>\n(2/7)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Sales Team Representative"),
                    KeyboardButton(text="Sales Manager"),
                    KeyboardButton(text="VIP Client Support Representative"),
                    KeyboardButton(text="Business Development Representative"),
                ],
            ],
            resize_keyboard=True,
        ),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.salesperson_role)
async def process_salesperson_role(message: Message, state: FSMContext) -> None:
    salesperson_role = message.text
    await state.update_data(salesperson_role=salesperson_role)
    await state.set_state(Processor.company_name)
    await message.answer(
        "<b>Please enter the company name or choose one below: </b>\n(3/7)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Reply.io"),
                    KeyboardButton(text="OutplayHQ"),
                    KeyboardButton(text="Amazon"),
                    KeyboardButton(text="Monobank"),
                ],
            ],
            resize_keyboard=True,
        ),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.company_name)
async def process_company_name(message: Message, state: FSMContext) -> None:
    company_name = message.text
    await state.update_data(company_name=company_name)
    await state.set_state(Processor.company_business)
    await message.answer(
        "<b>Please enter the company's business description or choose one below: </b>\n(4/7)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(
                        text="Our company is your AI-powered sales engagement platform to create new opportunities at scale – automatically."
                    ),
                    KeyboardButton(
                        text="Our company is a leading sales engagement platform."
                    ),
                    KeyboardButton(
                        text="Our company has been on the cutting edge of sales engagement technology since 1999."
                    ),
                    KeyboardButton(
                        text="Our company wants to connect businesses through personalized communication at scale."
                    ),
                ],
            ],
            resize_keyboard=True,
        ),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.company_business)
async def process_company_business(message: Message, state: FSMContext) -> None:
    company_business = message.text
    await state.update_data(company_business=company_business)
    await state.set_state(Processor.company_values)
    await message.answer(
        "<b>Please enter the company's core values and mission or choose below: </b>\n(5/7)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(
                        text="Our mission is to connect businesses through personalized communication at scale."
                    ),
                    KeyboardButton(
                        text="Our goal is helping the businesses to be more productive."
                    ),
                    KeyboardButton(
                        text="Our task is to change the way people work using AI."
                    ),
                    KeyboardButton(
                        text="Our idea is to make the world a better place by creating new opportunities."
                    ),
                ],
            ],
            resize_keyboard=True,
        ),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.company_values)
async def process_company_values(message: Message, state: FSMContext) -> None:
    company_values = message.text
    await state.update_data(company_values=company_values)
    await state.set_state(Processor.conversation_purpose)
    await message.answer(
        "<b>Please describe the purpose of this conversation or choose one below: </b>\n(6/7)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(
                        text="Answering the prospect's questions as a friendly sales manager."
                    ),
                    KeyboardButton(
                        text="Convincing the prospect to book a demo, proposing a trial, or sharing the contacts"
                    ),
                    KeyboardButton(
                        text="Forcing the prospect to book a demo, proposing a trial, or sharing the contacts as hard as possible"
                    ),
                    KeyboardButton(
                        text="Just having a friendly conversation about nature, history, or anything else."
                    ),
                ],
            ],
            resize_keyboard=True,
        ),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.conversation_purpose)
async def process_conversation_purpose(message: Message, state: FSMContext) -> None:
    conversation_purpose = message.text
    await state.update_data(conversation_purpose=conversation_purpose)
    await state.set_state(Processor.conversation_type)
    await message.answer(
        "<b>Please enter the conversation type or choose one below: </b>\n(7/7)",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Telegram Chat"),
                    KeyboardButton(text="Call"),
                    KeyboardButton(text="Email"),
                    KeyboardButton(text="Meeting"),
                ],
            ],
            resize_keyboard=True,
        ),
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
    # chat_bot.initialized = False
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
async def process_change_url_process(message: Message, state: FSMContext) -> None:
    """
    Handle the choice of what process to perform with the URL.
    """
    await state.update_data(path=message.text, is_file=False)
    await state.set_state(Processor.change_url_process)
    await message.answer(
        "What process do you want to perform with the URL?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Loader"),
                    KeyboardButton(text="Image Recognition"),
                ],
            ],
            resize_keyboard=True,
        ),
    )


@form_router.message(Processor.change_url_process, F.text.in_({"Loader", "Image Recognition"}))
async def process_change_url_process(message: Message, state: FSMContext) -> None:
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
        # f"Your url has been uploaded. What is your question?",
        f"Your url has been uploaded. Connecting the bot...",
        reply_markup=keyboard,
    )

    user_id = message.from_user.id
    data = await state.get_data()

    if user_id not in user_bots:
        args = dict(
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
            use_tools=True,
            is_file=data.get("is_file", False),
            path=data.get("path", ""),
        )

        user_bots[user_id] = SalesGPT.from_llm(
            # ChatOpenAI(temperature=0.8),
            # True,
            # **args
            ChatOpenAI(temperature=0.8, model="gpt-4"),
            True,
            **args,
        )

        user_bots[user_id].seed_agent()
        user_bots[user_id].step()

        output = (
            user_bots[user_id]
            .conversation_history[-1]
            .replace("<END_OF_TURN>", "")
            .strip()
        )

        await message.reply(output)

    await state.set_state(Processor.regular_usage)


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
            # f"Your file {file_name} has been uploaded. What is your question?",
            f"Your file {file_name} has been uploaded. Connecting the bot...",
            reply_markup=keyboard,
        )

        user_id = message.from_user.id
        data = await state.get_data()

        if user_id not in user_bots:
            args = dict(
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
                use_tools=True,
                is_file=data.get("is_file", False),
                path=data.get("path", ""),
            )

            user_bots[user_id] = SalesGPT.from_llm(
                ChatOpenAI(temperature=0.8),
                True,
                **args
                # ChatOpenAI(temperature=0.8, model="gpt-4"), True, **args
            )

            user_bots[user_id].seed_agent()
            user_bots[user_id].step()

            output = (
                user_bots[user_id]
                .conversation_history[-1]
                .replace("<END_OF_TURN>", "")
                .strip()
            )

            await message.reply(output)

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

    data = await state.get_data()
    print("regular usage", data.get("path"))
    keyboard = await create_regular_usage_keyboard()
    if not all([data.get("path")]):
        await message.answer(
            "Please, upload PDF file or url first.",
            reply_markup=keyboard,
        )
        return

    user_id = message.from_user.id

    user_bots[user_id].human_step(message.text)
    user_bots[user_id].step()

    output = (
        user_bots[user_id].conversation_history[-1].replace("<END_OF_TURN>", "").strip()
    )

    if "<END_OF_CALL>" in output:
        output = output.replace("<END_OF_CALL>", "")
        await message.reply(output)
        await state.set_data({})
        await state.set_state(Processor.regular_usage)

        keyboard = await create_regular_usage_keyboard()

        await message.answer(
            "Conversation finished. You may start a new conversation now.",
            reply_markup=keyboard,
        )
    else:
        await message.reply(output)


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
    conversation_type = data.get("conversation_type", "Telegram chat")

    text = f"""
        Your bot has been created with the following settings: 
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
