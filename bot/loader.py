from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from bot import config

bot = Bot(token=config.API_TOKEN,
          parse_mode=types.ParseMode.HTML)

dp = Dispatcher(bot, storage=MemoryStorage())



