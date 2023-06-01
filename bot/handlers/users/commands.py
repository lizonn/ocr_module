from aiogram import types
from aiogram.dispatcher.storage import FSMContext

from bot.handlers import texts
from bot.loader import dp
from bot.states.ocr_steps import Form
from bot.keyboards.main import main_kb,model_choice_kb


@dp.message_handler(commands="help", state="*")
async def help(message: types.Message, state: FSMContext):
    await message.answer(texts.HELP,reply_markup=main_kb)
    await state.reset_state(with_data=False)


@dp.message_handler(commands='image', state="*")
@dp.message_handler(text=texts.START_BUTTON)
async def image(message: types.Message, state: FSMContext):
    await message.answer(texts.CHOISE_MODEL)
    await message.answer(texts.CHOISE_MODEL2, reply_markup=model_choice_kb)

    await Form.waiting_for_model_choice.set()
