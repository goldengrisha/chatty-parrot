import asyncio
import logging
import sys

from os import getenv
from typing import Any, Dict

from aiogram import Bot, Dispatcher, F, Router, html
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from modules.settings import Settings


form_router = Router()


class Processor(StatesGroup):
    chat_bot_model = State()
    is_with_memory = State()
    is_with_context = State()
    is_with_internet_access = State()

    async def run(self):
        bot = Bot(token=Settings.get_tg_token(), parse_mode=ParseMode.HTML)
        dp = Dispatcher()
        dp.include_router(form_router)
        await dp.start_polling(bot)


@form_router.message(CommandStart())
async def command_start(message: Message, state: FSMContext) -> None:
    await state.set_state(Processor.chat_bot_model)
    await message.answer(
        f"Hi, I am a ContextGPT bot👋 \nPlease select chatbot type:",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="GPT-3.5"),
                    KeyboardButton(text="GPT-4"),
                ]
            ],
            resize_keyboard=True,
        ),
    )


@form_router.message(Command("help"))
async def command_help(message: Message):
    help_message = """
    Here are the available commands:
    /start - start the bot
    /reset - reset the context
    /cancel - cancel the current action
    /uploadPDF - upload new PDF file
    /sendURL - send new URL
    
    /help - show this help message
    """
    await message.answer(help_message, parse_mode=ParseMode.MARKDOWN)


@form_router.message(Processor.chat_bot_model, F.text.in_({"GPT-3.5", "GPT-4"}))
async def process_chat_bot_model(message: Message, state: FSMContext) -> None:
    await state.update_data(chat_bot_model=message.text)
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
        f"Would you like to use context instead of full chat gpt?",
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
        f"Would you like to use internet searching?",
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
    await state.clear()

    await show_summary(message=message, data=data)


@form_router.message(Command("cancel"))
@form_router.message(F.text.casefold() == "cancel")
async def cancel_handler(message: Message, state: FSMContext) -> None:
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()
    if current_state is None:
        return

    logging.info("Cancelling state %r", current_state)
    await state.clear()
    await message.answer(
        "Cancelled.",
        reply_markup=ReplyKeyboardRemove(),
    )


@form_router.message(Processor.chat_bot_model)
async def process_unknown_write_bots(message: Message) -> None:
    await message.reply("Choose the openai model firstly:")


@form_router.message(Processor.is_with_memory)
async def process_unknown_write_bots(message: Message) -> None:
    await message.reply("You need to choose whether to use memory in the bot:")


@form_router.message(Processor.is_with_context)
async def process_unknown_write_bots(message: Message) -> None:
    await message.reply("Choose using context instead of full chat gpt:")


@form_router.message(Processor.is_with_internet_access)
async def process_unknown_write_bots(message: Message) -> None:
    await message.reply("Do you want to use access to the internet?")


async def show_summary(
    message: Message, data: Dict[str, Any], positive: bool = True
) -> None:
    chat_bot_model = data.get("chat_bot_model", "GPT-3.5")
    is_with_memory = data.get("is_with_memory", "Yes")
    is_with_context = data.get("is_with_context", "Yes")
    is_with_internet_access = data.get("is_with_internet_access", "Yes")

    ai_key = Settings.get_ai_key(chat_bot_model)

    text = f"""
    Hey, your bot has been created! 
    Chatbot type: {chat_bot_model} 
    Memory: {is_with_memory} 
    Context: {is_with_context}  
    Internet access: {is_with_internet_access}"""

    await message.answer(text=text, reply_markup=ReplyKeyboardRemove())


@form_router.message(Command("reset"))
async def command_reset(message: Message) -> None:
    """
    Allow user to reset context
    """
    # here some logic
    await message.answer(
        "The context has been reset.",
        reply_markup=ReplyKeyboardRemove(),
    )


@form_router.message(Command("uploadPDF"))
async def command_upload_new_pdf(message: Message) -> None:
    """
    Allow user to upload new PDF file
    """
    await message.answer("Please choose the PDF file you want to upload.")

    # if message.document.mime_type == 'application/pdf':   not ended
    #     await message.answer("New PDF file has been uploaded.")

    # here must be process_pdf(update)


@form_router.message(Command("sendURL"))
async def command_send_new_url(message: Message) -> None:
    """
    Allow user to send new URL
    """
    # here some logic
    await message.answer(
        "New URL has been sent.",
        reply_markup=ReplyKeyboardRemove(),
    )

    # here must be process_url(update)
