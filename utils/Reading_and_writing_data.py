# Reading and writing data functions

# Получение размеров из изображений
def get_jpeg_size_pil(filename):
    with Image.open(filename) as img:
        width, height = img.size
        return width, height

# Сохранение Тензора в PNG
def save_tensor_as_png(tensor, filename):
    # Преобразование типа данных и масштабирование (если необходимо)
    if tensor.dtype == torch.uint8:
        # Если тензор уже в uint8, ничего не делаем с типом
        img_tensor = tensor.float() / 255.0 # Для PIL нужно, чтобы значения были от 0 до 1
    elif tensor.dtype == torch.float:
        # Проверяем, что значения в диапазоне [0, 1]
        if tensor.min() < 0 or tensor.max() > 1:
            print("Предупреждение: Значения тензора не в диапазоне [0, 1].  Масштабируем.")
            img_tensor = torch.clamp(tensor, 0, 1) #Обрезаем значения если выходят за границы
        else:
            img_tensor = tensor  # Не меняем, если уже от 0 до 1
    else:
        raise ValueError(f"Тип данных тензора не поддерживается: {tensor.dtype}. Используйте uint8 или float.")

    # Удаление размерности батча, если она есть
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor[0] # Берем первое изображение из батча

    # Изменение порядка размерностей (C, H, W) -> (H, W, C)
    img_tensor = img_tensor.permute(1, 2, 0)

    # Преобразование в NumPy array
    img_np = img_tensor.cpu().numpy()

    # Преобразование в PIL Image
    if img_np.shape[2] == 1:  # Grayscale
        img = Image.fromarray((img_np[:, :, 0] * 255).astype('uint8'), mode='L')
    elif img_np.shape[2] == 3:  # RGB
        img = Image.fromarray((img_np * 255).astype('uint8'), mode='RGB')
    elif img_np.shape[2] == 4:  # RGBA
        img = Image.fromarray((img_np * 255).astype('uint8'), mode='RGBA')
    else:
        raise ValueError(f"Неподдерживаемое количество каналов: {img_np.shape[2]}. Поддерживаются 1 (grayscale), 3 (RGB) или 4 (RGBA).")


    # Сохранение в PNG
    img.save(filename, "PNG")

