import aiohttp
from aiogram import types
from aiogram.dispatcher.storage import FSMContext
from matplotlib import pyplot as plt

from bot.handlers import texts
from bot.loader import dp,bot
from bot.states.ocr_steps import Form
from bot.keyboards.main import model_choice_kb
from bot.middlewares.check_image import image_quality_check

import io
import cv2
import numpy as np
from PIL import Image
from bot import config
from ocr_module.main import get_text_from_image

# @dp.message_handler(text=texts.START_BUTTON)
# async def waiting_for_model_choice(message: types.Message, state: FSMContext):
#     await message.answer(texts.CHOISE_MODEL,reply_markup=types.ReplyKeyboardRemove())
#     await message.answer(texts.CHOISE_MODEL2,reply_markup=model_choice_kb)
#
#     await Form.waiting_for_model_choice.set()



VALID_MODELS = [texts.MODEL1,texts.MODEL2]

@dp.message_handler(state=Form.waiting_for_model_choice)
async def process_model_choice(message: types.Message, state: FSMContext):
    chosen_model = message.text

    data = await state.get_data()
    attempts = data.get('attempts', 0)

    if chosen_model not in VALID_MODELS:
        attempts += 1
        await state.update_data(attempts=attempts)
        if attempts >= 2:
            chosen_model = VALID_MODELS[1]  # автоматически выбирается модель 2
            await message.answer('Ви не правильно ввели модель. Було автоматично вибрано друга модель.',reply_markup=types.ReplyKeyboardRemove())
        else:
            await message.answer('Ви ввели неправильну модель. Будь ласка, введіть назву моделі ще раз.',reply_markup=model_choice_kb)
            return
    else:
        await message.answer(
            'Ви вибрали ' + chosen_model + '. Будь ласка, завантажте зображення високої якості, на якому ви хочете розпізнати текст.',reply_markup=types.ReplyKeyboardRemove())

    await state.update_data(chosen_model=chosen_model)
    await Form.waiting_for_image.set()


@dp.message_handler(content_types=['text', 'photo', 'document'], state=Form.waiting_for_image)
@image_quality_check
async def process_image(message: types.Message, state: FSMContext, image_path=None, **kwargs):

    await message.answer(texts.WAIT_MODEL)
    data = await state.get_data()
    chosen_model = data.get('chosen_model')


    # TODO можна винести потім в окрему функцію
    file_path = await bot.get_file(image_path)
    file_url = f"https://api.telegram.org/file/bot{config.API_TOKEN}/{file_path.file_path}"

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as resp:
            file_body = await resp.read()

    image = Image.open(io.BytesIO(file_body))

    if chosen_model == texts.MODEL1:
        model_name = 'ocr_module/models/ua_model.h5'
    else:
        model_name = 'ocr_module/models/ua_model.h5'

    res = get_text_from_image(model_name, image)
    await message.answer(f'Ваш розпізнаний текст:\n{res}')

    await state.finish()
