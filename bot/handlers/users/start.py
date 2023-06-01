from aiogram import types
from aiogram.dispatcher.filters.builtin import CommandStart
from bot.loader import dp
from bot.handlers import texts

from bot.keyboards.main import main_kb

@dp.message_handler(CommandStart(),state='*')
async def start_with_deeplink(message: types.Message,state):

    await state.reset_state(with_data=False)

    await message.answer(texts.GREETING_TEXT,reply_markup=main_kb)

