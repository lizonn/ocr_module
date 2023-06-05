
from aiogram.dispatcher.middlewares import BaseMiddleware
from aiogram import types, Dispatcher
from aiogram.dispatcher.storage import FSMContext
from bot.loader import bot
from PIL import Image

from io import BytesIO
from PIL import Image

QUALITY_THRESHOLD = 1000000  # тестове знчення

async def check_image_quality(file: bytes):
    image = Image.open(BytesIO(file))
    width, height = image.size
    return width * height >= QUALITY_THRESHOLD

def image_quality_check(function):
    async def wrapper(message: types.Message, state: FSMContext, *args, **kwargs):
        if message.content_type in ['photo', 'document']:
            if message.content_type == 'document' and not message.document.mime_type.startswith('image/'):
                await message.answer('Будь ласка, завантажте зображення')
            else:
                # ПОКИ ЩО НЕ ПРАЦЮЄ З ФАЙЛАМИ

                return await function(message, state, image_path=message.photo[-1].file_id, *args, **kwargs)

                # file_id = message.photo[-1].file_id if message.content_type == 'photo' else message.document.file_id
                # file = await bot.download_file_by_id(file_id)
                # if await check_image_quality(file):
                #     return await function(message, state, image_file=file, *args, **kwargs)
                # else:
                #     # await message.answer('5')
                #     await message.answer('Якість зображення недостатньо хороша. Будь ласка, завантажте інше зображення')
        else:
            await message.answer('Будь ласка, завантажте зображення')
    return wrapper
