import asyncio
import logging
import sys

from os import getenv
from typing import Any, Dict

import aiofiles

from aiogram import Bot, Dispatcher, F, Router, html
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from urllib.parse import urlparse

from modules.chat_bot import ChatBot

# from aiogram.dispatcher.filters.builtin import ContentTypesFilter
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
chat_bot = ChatBot()


class Processor(StatesGroup):
    chat_bot_type = State()
    is_with_memory = State()
    is_with_context = State()
    is_with_internet_access = State()
    regular_usage = State()
    change_context = State()
    waiting_for_file = State()
    waiting_for_url = State()

    async def run(self):
        global bot
        global dp
        dp.include_router(form_router)
        await dp.start_polling(bot)


@form_router.message(CommandStart())
async def command_start(message: Message, state: FSMContext) -> None:
    await state.set_state(Processor.chat_bot_type)
    await message.answer(
        f"Please select chatbot type.",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="GPT-3.5-Turbo"),
                    KeyboardButton(text="GPT-4"),
                ]
            ],
            resize_keyboard=True,
        ),
    )


@form_router.message(Processor.chat_bot_type, F.text.in_({"GPT-3.5-Turbo", "GPT-4"}))
async def process_chat_bot_type(message: Message, state: FSMContext) -> None:
    await state.update_data(chat_bot_type=message.text)
    await state.set_state(Processor.is_with_memory)
    await message.answer(
        f"Would you like to use memory?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Yes"),
                    KeyboardButton(text="No"),
                ]
            ],
            resize_keyboard=True,
        ),
    )


@form_router.message(Processor.is_with_memory, F.text.in_({"Yes", "No"}))
async def process_memory(message: Message, state: FSMContext) -> None:
    await state.update_data(is_with_memory=message.text)
    await state.set_state(Processor.is_with_context)
    await message.answer(
        f"Would you like to use context or full chat gpt?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Yes"),
                    KeyboardButton(text="No"),
                ]
            ],
            resize_keyboard=True,
        ),
    )


@form_router.message(Processor.is_with_context, F.text.in_({"Yes", "No"}))
async def process_context(message: Message, state: FSMContext) -> None:
    await state.update_data(is_with_context=message.text)
    await state.set_state(Processor.is_with_internet_access)
    await message.answer(
        f"Would you like to use internet access?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="Yes"),
                    KeyboardButton(text="No"),
                ]
            ],
            resize_keyboard=True,
        ),
    )


@form_router.message(
    Processor.is_with_internet_access,
    F.text.in_({"Yes", "No"}),
)
async def process_internet_access(message: Message, state: FSMContext) -> None:
    data = await state.update_data(is_with_internet_access=message.text)
    await state.set_state(Processor.regular_usage)

    await show_summary(message=message, data=data, keyboard=ReplyKeyboardRemove())
    await message.answer(
        "You can use your bot now.",
        reply_markup=ReplyKeyboardMarkup(
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
        ),
    )


@form_router.message(Processor.regular_usage, F.text.in_({"Show status"}))
async def process_regular_usage_show_status(
    message: Message, state: FSMContext
) -> None:
    await show_summary(
        message=message,
        data=await state.get_data(),
        keyboard=ReplyKeyboardMarkup(
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
        ),
    )


@form_router.message(Processor.regular_usage, F.text.in_({"Reset", "/reset"}))
async def process_regular_usage_reset(message: Message, state: FSMContext) -> None:
    chat_bot.initialized = False
    await state.set_data({})
    await state.set_state(Processor.chat_bot_type)
    await message.answer("Let's configure your bot again.")
    await message.answer(
        f"Please select chatbot type.",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="GPT-3.5-Turbo"),
                    KeyboardButton(text="GPT-4"),
                ]
            ],
            resize_keyboard=True,
        ),
    )


@form_router.message(Processor.regular_usage, F.text.casefold() == "/uploadpdf")
async def process_uploadPDF(message: Message, state: FSMContext) -> None:
    await state.set_state(Processor.waiting_for_file)
    await message.answer("Please, upload PDF file.", reply_markup=ReplyKeyboardRemove())


@form_router.message(Processor.regular_usage, F.text.casefold() == "/sendurl")
async def process_uploadPDF(message: Message, state: FSMContext) -> None:
    await state.set_state(Processor.waiting_for_url)
    await message.answer("Please, paste url here.", reply_markup=ReplyKeyboardRemove())


@form_router.message(Processor.regular_usage, F.text.in_({"Upload PDF file or URL"}))
async def process_regular_usage_document(message: Message, state: FSMContext) -> None:
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
    await state.set_state(Processor.waiting_for_file)
    await message.answer("Please, upload a pdf file.", reply_markup=ReplyKeyboardRemove())


@form_router.message(Processor.change_context, F.text.in_({"Url"}))
async def process_change_context_document(message: Message, state: FSMContext) -> None:
    await state.set_state(Processor.waiting_for_url)
    await message.answer("Please, paste url here.", reply_markup=ReplyKeyboardRemove())


@form_router.message(Processor.waiting_for_url)
async def process_waiting_for_url(message: Message, state: FSMContext) -> None:
    url = message.text

    # Check if it's a valid URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        await message.answer("Please paste a valid URL.")
        return
    await state.update_data(path=message.text, is_file=False)
    await state.set_state(Processor.regular_usage)
    await message.answer(
        f"Your url has been uploaded.",
        reply_markup=ReplyKeyboardMarkup(
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
        ),
    )


@form_router.message(Processor.waiting_for_url)
def process_invalid_url(message: Message, state: FSMContext) -> None:
    message.answer("Invalid url. Please paste here valid URL.")


@form_router.message(Processor.waiting_for_file, F.document)
async def process_waiting_for_file(message: Message, state: FSMContext) -> None:
    try:
        # Download the document
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path
        if not file_path.endswith(".pdf"):
            await message.answer(
                "Please upload a PDF file."
            )
        download_file = await bot.download_file(file_path)
        async with aiofiles.open(
            f"downloads/{message.document.file_name}", mode="wb"
        ) as file:
            await file.write(download_file.read())
        await state.update_data(
            path=f"downloads/{message.document.file_name}", is_file=True
        )
        await message.answer(
            f"Your file {message.document.file_name} has been uploaded.",
            reply_markup=ReplyKeyboardMarkup(
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
            ),
        )
        await state.set_state(Processor.regular_usage)

    except Exception as e:
        await message.answer(f"An error occurred: {e}")


@form_router.message(Processor.waiting_for_file)
def process_invalid_file(message: Message, state: FSMContext) -> None:
    message.answer("Please upload a file.")


# @form_router.message(Command("cancel"))
# @form_router.message(F.text.casefold() == "cancel")
# async def cancel_handler(message: Message, state: FSMContext) -> None:
#     """
#     Allow user to cancel any action
#     """
#     current_state = await state.get_state()
#     if current_state is None:
#         return

#     logging.info("Cancelling state %r", current_state)
#     await state.clear()
#     await message.answer(
#         "Cancelled.",
#         reply_markup=ReplyKeyboardRemove(),
#     )


@form_router.message(Processor.regular_usage, F.text.casefold() == "/help")
async def process_regular_usage_reset(message: Message, state: FSMContext) -> None:
    await print_help(message)


@form_router.message(Processor.regular_usage)
async def process_regular_usage_reset(message: Message, state: FSMContext) -> None:
    global chat_bot
    data = await state.get_data()
    print("data", data)
    if not all([data.get("path")]):
        await message.answer(
            "Please, upload PDF file or url first.",
            reply_markup=ReplyKeyboardMarkup(
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
            ),
        )
        return
    if not chat_bot.initialized:
        chat_bot.initialize(
            data.get("chat_bot_type").lower(),
            data.get("is_with_memory") == "Yes",
            data.get("is_with_context") == "Yes",
            data.get("is_with_internet_access") == "Yes",
            data.get("is_file"),
            data.get("path")
        )
    
    if chat_bot.path != data.get("path") or chat_bot.is_file != data.get("is_file"):
        chat_bot.initialize(
            data.get("chat_bot_type").lower(),
            data.get("is_with_memory") == "Yes",
            data.get("is_with_context") == "Yes",
            data.get("is_with_internet_access") == "Yes",
            data.get("is_file"),
            data.get("path")
        )
    answer = chat_bot.query_executor.invoke(message.text)
    await message.reply(
        answer["output"]
    )


@form_router.message(Processor.chat_bot_type)
async def process_unknown_write_bots(message: Message) -> None:
    await message.reply("I don't understand you :(")


@form_router.message(Processor.is_with_memory)
async def process_unknown_write_bots(message: Message) -> None:
    await message.reply("I don't understand you :(")


@form_router.message(Processor.is_with_context)
async def process_unknown_write_bots(message: Message) -> None:
    await message.reply("I don't understand you :(")


@form_router.message(Processor.is_with_internet_access)
async def process_unknown_write_bots(message: Message) -> None:
    await message.reply("I don't understand you :(")


async def show_summary(
    message: Message,
    data: Dict[str, Any],
    keyboard,
    positive: bool = True,
) -> None:
    chat_bot_type = data.get("chat_bot_type", "None")
    is_with_memory = data.get("is_with_memory", "None")
    is_with_context = data.get("is_with_context", "None")
    is_with_internet_access = data.get("is_with_internet_access", "None")

    ai_key = Settings.get_ai_key(chat_bot_type)

    text = f"""
    Hey, your bot has been created! 
    Chatbot type: {chat_bot_type} 
    Memory: {is_with_memory} 
    Context: {is_with_context}  
    Internet access: {is_with_internet_access}"""

    await message.answer(text=text, reply_markup=keyboard)


@form_router.message(F.text.casefold() == "/help")
async def command_help(message: Message, state: FSMContext):
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
    help_message = """
    Here are the available commands:
    /start - start the bot
    /reset - reset the context
    /uploadPDF - upload new PDF file
    /sendURL - send new URL
    
    /help - show this help message
    """
    await message.answer(help_message, parse_mode=ParseMode.MARKDOWN)
