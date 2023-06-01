from bot.handlers import dp
from bot.loader import bot

from aiogram.types import (BotCommand,
                           BotCommandScopeAllPrivateChats)

from aiogram import executor


def start_bot(dp):
    user_commands = [
        BotCommand('image', 'Розпізнати нове зображення'),
        BotCommand('help', 'Інструкція як користуватися ботом')
    ]

    async def startup(dp):
        await bot.set_my_commands(
            commands=user_commands,
            scope=BotCommandScopeAllPrivateChats())

    executor.start_polling(dp, on_startup=startup)


if __name__ == '__main__':
    start_bot(dp)
