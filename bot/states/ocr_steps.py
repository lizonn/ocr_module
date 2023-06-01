from aiogram.dispatcher.filters.state import StatesGroup, State


class Form(StatesGroup):
    waiting_for_model_choice = State()
    waiting_for_image = State()
