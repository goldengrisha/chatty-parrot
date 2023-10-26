import aiofiles  # type: ignore
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

from modules.enum import SalesBotResponseSize
from modules.sales_chat_bot import SalesGPT

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


def get_tone_instruction(salesperson_tone):
    """
    Returns a detailed and strict instruction for the given salesperson tone.

    Parameters:
    - salesperson_tone (str): The tone of the salesperson.

    Returns:
    - str: Detailed instruction for the given tone.
    """

    tone_instructions = {
        "Formal and Professional": (
            "Maintain a polished and businesslike demeanor in your responses."
            "Use complete sentences, avoid contractions (e.g., use 'cannot' instead of 'can't')."
            "If you don't know the prospect title (e.g., Mr., Mrs., Dr.), you must specifically ask how to address them in the first message."
            "Ensure to address them accordingly."
        ),
        "Conversational and Friendly": (
            "Aim for a casual and warm tone."
            "Imagine you're speaking to a close friend."
            "Use contractions (e.g., 'I'm', 'can't'), and feel free to sprinkle in colloquial expressions or idioms where appropriate."
        ),
        "Inspirational and Motivational": (
            "Infuse your messages with positivity and encouragement."
            "Utilize uplifting words and phrases, and always focus on the positive aspects of any situation or product feature."
        ),
        "Empathetic and Supportive": (
            "Always show understanding and compassion."
            "Put yourself in the prospect's shoes and validate their feelings or concerns."
            "Use phrases like 'I understand how you feel' or 'That must be tough'."
        ),
        "Educational and Informative": (
            "Prioritize providing clear and valuable information."
            "Break down complex concepts into simpler terms, and always be prepared to offer further explanation or examples."
            "Your should educate the prospect."
        ),
        "Neutral": (
            "Maintain a balanced and unbiased tone."
            "Avoid showing excessive emotion or bias in any direction."
            "Respond to the prospect's inquiries in a straightforward manner without leaning too much towards any specific emotion or style."
        ),
    }

    return tone_instructions.get(
        salesperson_tone,
        (
            "Maintain a balanced and unbiased tone."
            "Avoid showing excessive emotion or bias in any direction."
            "Respond to the prospect's inquiries in a straightforward manner without leaning too much towards any specific emotion or style."
        ),
    )


def get_language_instruction(salesperson_language):
    """
    Returns a detailed and strict instruction for the given salesperson language.

    Parameters:
    - salesperson_language (str): The language of the salesperson.

    Returns:
    - str: Detailed instruction for the given language.
    """

    tone_instructions = {
        "English": (
            "Everything you say must be in English and English only."
            "No other languages are allowed for you. If the user speaks different language, you must still answer in English."
        ),
        "Dynamic": (
            "You must follow the user's language and answer in the same language."
            "You are not allowed to answer in any other language except for the language the user speaks to you."
            "The user must be able to understand you at all times so be sure to use the same language as they do."
        ),
    }

    return tone_instructions.get(
        salesperson_language,
        (
            "Everything you say must be in English and English only."
            "No other languages are allowed for you. If the user speaks different language, you must still answer in English."
        ),
    )


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
    salesperson_language = State()
    salesperson_role = State()
    salesperson_tone = State()
    context_restriction = State()
    company_name = State()
    company_business = State()
    company_values = State()
    conversation_purpose = State()
    conversation_type = State()
    conversation_stage = State()
    salesperson_response_size = State()
    conversation_language = State()

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
        "<b>Please enter the bot's name or choose one below: </b>",
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
    await state.set_state(Processor.salesperson_language)

    keyboard = await create_language_keyboard()

    await message.answer(
        "<b>Please choose the bot's language below: </b>\nEnglish – use only English\nDynamic – use the user's language",
        reply_markup=keyboard,
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.salesperson_language)
async def process_salesperson_language(message: Message, state: FSMContext) -> None:
    salesperson_language = message.text

    await state.update_data(salesperson_language=salesperson_language)
    await state.update_data(
        salesperson_language_instruction=get_language_instruction(salesperson_language)
    )

    await state.set_state(Processor.context_restriction)
    await message.answer(
        "<b>Should the bot be restricted by the context oly, or can it use ChatGPT knowledge as well: </b>",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Context only"),
                    KeyboardButton(text="ChatGPT knowledge"),
                ],
            ],
            resize_keyboard=True,
        ),
        parse_mode=ParseMode.HTML,
    )


context_temperature = 0.0


@form_router.message(Processor.context_restriction)
async def process_context_restriction(message: Message, state: FSMContext) -> None:
    global context_temperature

    context_restriction = message.text
    context_temperature = 0.0 if context_restriction == "Context only" else 0.8

    await state.update_data(context_restriction=context_restriction)

    await state.set_state(Processor.salesperson_role)
    await message.answer(
        "<b>Please enter the bot's role or choose one below: </b>",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Sales Representative"),
                    KeyboardButton(text="Sales Manager"),
                    KeyboardButton(text="Account Executive"),
                    KeyboardButton(text="Business Development Manager"),
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
    await state.set_state(Processor.salesperson_tone)
    await message.answer(
        "<b>Please choose the bot's tone: </b>",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Neutral"),
                    KeyboardButton(text="Formal and Professional"),
                    KeyboardButton(text="Conversational and Friendly"),
                    KeyboardButton(text="Inspirational and Motivational"),
                    KeyboardButton(text="Empathetic and Supportive"),
                    KeyboardButton(text="Educational and Informative"),
                ],
            ],
            resize_keyboard=True,
        ),
        parse_mode=ParseMode.HTML,
    )


@form_router.message(Processor.salesperson_tone)
async def process_salesperson_tone(message: Message, state: FSMContext) -> None:
    salesperson_tone = message.text
    await state.update_data(salesperson_tone=get_tone_instruction(salesperson_tone))
    await state.set_state(Processor.company_name)
    await message.answer(
        "<b>Please enter the company name or choose one below: </b>",
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
        "<b>Please enter the company's business description or choose one below: </b>",
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
        "<b>Please enter the company's core values and mission or choose below: </b>",
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
        "<b>Please describe the purpose of this conversation or choose one below: </b>",
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
    await state.set_state(Processor.salesperson_response_size)

    keyboard = await create_salesperson_response_size()

    await message.answer(
        "Which response size would you prefer?",
        reply_markup=keyboard,
    )


@form_router.message(Processor.salesperson_response_size)
async def process_salesperson_response_size(
    message: Message, state: FSMContext
) -> None:
    selected_size = message.text
    if selected_size == "SMALL":
        salesperson_response_size = SalesBotResponseSize.SMALL.value
    elif selected_size == "MEDIUM":
        salesperson_response_size = SalesBotResponseSize.MEDIUM.value
    elif selected_size == "LARGE":
        salesperson_response_size = SalesBotResponseSize.LARGE.value
    data = await state.update_data(salesperson_response_size=salesperson_response_size)
    await state.set_state(Processor.regular_usage)

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
    await state.set_data({})
    await state.set_state(Processor.salesperson_name)
    await message.answer("Let's configure your bot again!")
    await message.answer(
        "<b>Please enter the bot's name or choose one below: </b>\n(1/8)",
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
        f"Would you like upload PDF file or URL?",
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


@form_router.message(  # type: ignore
    Processor.change_url_process, F.text.in_({"Loader", "Image Recognition"})
)
async def process_change_url_process(message: Message, state: FSMContext) -> None:
    url_process = message.text
    await state.update_data(url_process=url_process)
    await state.set_state(Processor.processing_url)
    await message.answer(
        f"You have selected {url_process}. Please, provide the URL.",
        reply_markup=ReplyKeyboardRemove(),
    )


@form_router.message(Processor.processing_url)
async def process_processing_url(message: Message, state: FSMContext) -> None:
    """
    Handle the URL processing, e.g., web page loading and image recognition.
    """
    url = message.text

    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        await message.answer("Please, provide a valid URL.")
        return
    url_process = (await state.get_data())["url_process"]
    await state.update_data(path=message.text, is_file=False, url_process=url_process)
    await state.set_state(Processor.regular_usage)
    keyboard = await create_regular_usage_keyboard()

    await message.answer(
        f"Your URL has been uploaded. Connecting the bot...",
        reply_markup=keyboard,
    )

    user_id = message.from_user.id
    data = await state.get_data()

    args = dict(
        salesperson_name=data.get("salesperson_name", "John"),
        salesperson_language=data.get("salesperson_language", "English"),
        salesperson_role=data.get(
            "salesperson_role", "Business Development Representative"
        ),
        salesperson_tone=data.get(
            "salesperson_tone",
            (
                "Maintain a balanced and unbiased tone."
                "Avoid showing excessive emotion or bias in any direction."
                "Respond to the prospect's inquiries in a straightforward manner without leaning too much towards any specific emotion or style."
            ),
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
        conversation_stage=(
            "Introduction: Start the conversation by introducing yourself and your company.",
            "Be polite and respectful while keeping the tone of the conversation professional.",
        ),
        use_tools=True,
        file_type=data.get("is_file", False),
        path=data.get("path", ""),
        url_loading_type=data.get("url_process", "Loader"),
        salesperson_response_size=data.get("salesperson_response_size", "Medium"),
    )

    user_bots[user_id] = SalesGPT.from_llm(
        ChatOpenAI(temperature=context_temperature, model="gpt-4"),
        True,
        **args,
    )

    user_bots[user_id].seed_agent()
    user_bots[user_id].step()

    output = (
        user_bots[user_id].conversation_history[-1].replace("<END_OF_TURN>", "").strip()
    )

    await message.reply(output)

    await state.set_state(Processor.regular_usage)


@form_router.message(Processor.waiting_for_url)
def process_invalid_url(message: Message, state: FSMContext) -> None:
    """
    Handle the case where the user enters an invalid URL.
    """
    message.answer("Invalid URL. Please provide a valid URL.")


@form_router.message(Processor.waiting_for_file, F.document)
async def process_waiting_for_file(message: Message, state: FSMContext) -> None:
    """
    Handle the user uploading a document.
    """
    try:
        if not os.path.exists("downloads"):
            os.makedirs("downloads")
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path
        if not file_path.endswith(".pdf"):
            await message.answer("Please upload a PDF file.")
        download_file = await bot.download_file(file_path)
        file_name = message.document.file_name
        local_path = f"downloads/{file_name}"

        async with aiofiles.open(local_path, mode="wb") as local_file:
            await local_file.write(download_file.read())

        await state.update_data(path=local_path, is_file=True)
        keyboard = await create_regular_usage_keyboard()

        await message.answer(
            f"Your file {file_name} has been uploaded. Connecting the bot...",
            reply_markup=keyboard,
        )

        user_id = message.from_user.id
        data = await state.get_data()

        if user_id not in user_bots:
            args = dict(
                salesperson_name=data.get("salesperson_name", "John"),
                salesperson_language=data.get("salesperson_language", "English"),
                salesperson_role=data.get(
                    "salesperson_role", "Business Development Representative"
                ),
                salesperson_tone=data.get(
                    "salesperson_tone",
                    (
                        "Maintain a balanced and unbiased tone."
                        "Avoid showing excessive emotion or bias in any direction."
                        "Respond to the prospect's inquiries in a straightforward manner without leaning too much towards any specific emotion or style."
                    ),
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
                conversation_stage=(
                    "Introduction: Start the conversation by introducing yourself and your company.",
                    "Be polite and respectful while keeping the tone of the conversation professional.",
                ),
                use_tools=True,
                file_type=data.get("is_file", False),
                path=data.get("path", ""),
                url_loading_type=data.get("url_process", "Loader"),
                salesperson_response_size=data.get(
                    "salesperson_response_size", "SMALL"
                ),
            )

            user_bots[user_id] = SalesGPT.from_llm(
                ChatOpenAI(temperature=context_temperature), True, **args
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


@form_router.message(Processor.regular_usage, F.text.casefold() == "/help")  # type: ignore
async def process_regular_usage_reset(message: Message, state: FSMContext) -> None:
    """
    Handle the command to display the help message.
    """
    await print_help(message)


@form_router.message(Processor.regular_usage)  # type: ignore
async def process_regular_usage_reset(message: Message, state: FSMContext) -> None:
    """
    Handle regular usage of the bot, processing user queries.
    """

    data = await state.get_data()
    print("regular usage", data.get("path"))
    keyboard = await create_regular_usage_keyboard()
    if not all([data.get("path")]):
        await message.answer(
            "Please, upload PDF file or URL first.",
            reply_markup=keyboard,
        )
        return

    user_id = message.from_user.id
    user_message = message.text

    user_bots[user_id].human_step(user_message)
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
    salesperson_language = data.get("salesperson_language", "English")
    salesperson_role = data.get(
        "salesperson_role", "Business Development Representative"
    )
    salesperson_tone = data.get(
        "salesperson_tone",
        (
            "Maintain a balanced and unbiased tone."
            "Avoid showing excessive emotion or bias in any direction."
            "Respond to the prospect's inquiries in a straightforward manner without leaning too much towards any specific emotion or style."
        ),
    )
    context_restriction = data.get("context_restriction", "Context only")
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

    text = f"""
        Your bot has been created with the following settings: 
        Chatbot name: {salesperson_name} 
        Chatbot language: {salesperson_language} 
        Chatbot role: {salesperson_role} 
        Chatbot tone: {salesperson_tone} 
        Company name: {company_name}
        Context restricted: {context_restriction}
        Company business: {company_business}
        Company values: {company_values}
        Conversation purpose: {conversation_purpose}"""

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


async def create_language_keyboard():
    """
    Create a custom keyboard for language.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="English"),
                KeyboardButton(text="Dynamic"),
            ],
        ],
        resize_keyboard=True,
    )


async def create_salesperson_response_size():
    """
    Create a custom keyboard for size of response.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="SMALL"),
                KeyboardButton(text="MEDIUM"),
                KeyboardButton(text="LARGE"),
            ],
        ],
        resize_keyboard=True,
    )
