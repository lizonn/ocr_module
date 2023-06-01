from aiogram.types import ReplyKeyboardMarkup,  KeyboardButton
from bot.handlers import texts



b1 = KeyboardButton(texts.START_BUTTON)


main_kb = ReplyKeyboardMarkup(resize_keyboard=True)
main_kb.add(b1)


model1 = KeyboardButton(texts.MODEL1)
model2 = KeyboardButton(texts.MODEL2)

model_choice_kb = ReplyKeyboardMarkup(resize_keyboard=True)
model_choice_kb.add(model1)
model_choice_kb.add(model2)