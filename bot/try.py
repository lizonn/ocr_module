



@dp.message_handler(state=Form.waiting_for_image, content_types=types.ContentType.PHOTO)
async def process_image(message: types.Message, state: FSMContext):
    await state.update_data(image=message.photo[-1].file_id)
    await message.answer('Выберите модель для распознавания текста: модель 1 или модель 2.')
    await Form.waiting_for_model_choice.set()

@dp.message_handler(state=Form.waiting_for_model_choice)
async def process_model_choice(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    image = user_data['image']
    model_choice = message.text
    # Proceed with OCR using the chosen model here
    await state.finish()