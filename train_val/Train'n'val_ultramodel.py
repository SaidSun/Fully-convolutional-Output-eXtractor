# Train'n'val ultramodel

def training_only_yolo(model, i, epoch, batch, loss_func, len_train_loader, logging_wandb, is_last=False, last_loss=None):
    running_loss = 0.0
    # Чтение данных
    batch["images"] = normalize_pad(batch['images'].float().squeeze(1).to(device))
    print("Image data: ", "Min: ", torch.min(batch["images"]), "Max: ", torch.max(batch["images"]))
    images = batch['images'].requires_grad_(True)
    boxes = batch['boxes'] 
    classes = batch['clss']

    # Выпрямление боксов и классов в один вектор
    flat_boxes = torch.stack([box for sublist in boxes for box in sublist]).float().requires_grad_(True)
    flat_classes = torch.tensor([clss for sublist in classes for clss in sublist]).float().requires_grad_(True)
    print("Box data: ", "Min: ", torch.min(flat_boxes), "Max: ", torch.max(flat_boxes))
    print("Class data: ", "Min: ", torch.min(flat_classes), "Max: ", torch.max(flat_classes))
    
    # Генераация batch_idx
    batch_idx = []
    for j, sublist in enumerate(boxes):
        batch_idx.extend(torch.tensor([j] * len(sublist)))
    batch_idx = torch.tensor(batch_idx).float().requires_grad_(True)


    # Вывод модели
    output = model(images[:, :3, :, :])

    # Перегруппировка таргетов для правильного формата
    target = {"batch_idx" : batch_idx, "cls" : flat_classes, "bboxes" : flat_boxes}

    # Рассчет ошибки
    model_loss = loss_func(output, target)

    #Преобразования выхода в читаемый формат
    # results = preprocess_data(output, target)

    # Усреднение ошибки
    model_loss = model_loss.mean()
    print("My model loss: ", model_loss)
    running_loss += model_loss.item()

    #Обработка ошибки при обучении
    optimizer.zero_grad()
    model_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.yolo.parameters(), max_norm=1.)
    model.check_grads(logging_wandb, running_loss)
    model.logging_grads_info(logging_wandb)
    optimizer.step()
    
    
    # if is_last:
    #     loss_func.result += last_loss
    loss_func.logging_loss("train")

    print(f"TRAIN: Epoch [{epoch+1}], Batch [{i+1}/{len_train_loader}], Loss: {running_loss}")
    
    # return results
    return model_loss

def val_only_yolo(model, i, epoch, batch, loss_func, len_val_loader, logging_wandb, is_last=False):
    running_loss = 0.0
    # Чтение данных
    batch["images"] = normalize_pad(batch['images'].float().squeeze(1).to(device))
    print("Image data: ", "Min: ", torch.min(batch["images"]), "Max: ", torch.max(batch["images"]))
    images = batch['images']
    boxes = batch['boxes'] 
    classes = batch['clss']

    # Выпрямление боксов и классов в один вектор
    flat_boxes = torch.stack([box for sublist in boxes for box in sublist]).float()
    flat_classes = torch.tensor([clss for sublist in classes for clss in sublist]).float()
    print("Box data: ", "Min: ", torch.min(flat_boxes), "Max: ", torch.max(flat_boxes))
    print("Class data: ", "Min: ", torch.min(flat_classes), "Max: ", torch.max(flat_classes))
    
    # Генераация batch_idx
    batch_idx = []
    for j, sublist in enumerate(boxes):
        batch_idx.extend(torch.tensor([j] * len(sublist)))
    batch_idx = torch.tensor(batch_idx).float()

    # Генерация вывода обертки модели (дял проверки)

    # Вывод модели
    output = model(images[:, :3, :, :])

    # Перегруппировка таргетов для правильного формата
    target = {"batch_idx" : batch_idx, "cls" : flat_classes, "bboxes" : flat_boxes}

    # Рассчет ошибки
    model_loss = loss_func(output, target)

    #Преобразования выхода в читаемый формат
    # results = preprocess_data(output, target)

    # Усреднение ошибки
    model_loss = model_loss.mean()
    print("My model loss: ", model_loss)
    running_loss = model_loss.item()
    
    if is_last:
        loss_func.result /= len_val_loader
        loss_func.logging_loss("val")

        print(f"VAL: Epoch [{epoch+1}], Batch [{i+1}/{len_val_loader}], Loss: {running_loss/len_val_loader}")
    
    # return results
    return None    
    

def training_with_YoloLoss(model, i, epoch, batch, loss_func, len_train_loader, logging_wandb, is_last=False, last_loss=None):
    running_loss = 0.0
    # Чтение данных
    batch["images"] = normalize_pad(batch['images'].float().squeeze(1).to(device))
    images = batch['images'].requires_grad_(True)
    boxes = batch['boxes'] 
    classes = batch['clss']

    # Выпрямление боксов и классов в один вектор
    flat_boxes = torch.stack([box for sublist in boxes for box in sublist]).float().requires_grad_(True)
    flat_classes = torch.tensor([clss for sublist in classes for clss in sublist]).float().requires_grad_(True)
    
    # Генераация batch_idx
    batch_idx = []
    for j, sublist in enumerate(boxes):
        batch_idx.extend(torch.tensor([j] * len(sublist)))
    batch_idx = torch.tensor(batch_idx).float().requires_grad_(True)

    # Вывод модели
    output = model(images)

    # Перегруппировка таргетов для правильного формата
    target = {"batch_idx" : batch_idx, "cls" : flat_classes, "bboxes" : flat_boxes}

    # Рассчет ошибки
    model_loss = loss_func(output, target)
    
    
    #Преобразования выхода в читаемый формат
    # results = preprocess_data(output, target)

    # Усреднение ошибки
    model_loss = model_loss.mean()
    print("My model loss: ", model_loss)
    running_loss += model_loss.item()
    
    #Обработка ошибки при обучении
    optimizer.zero_grad()
    model_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.yolo.parameters(), max_norm=1.)
    model.check_grads(logging_wandb, running_loss)
    model.logging_grads_info(logging_wandb)
    optimizer.step()
    
    # if is_last:
        # loss_func.result += last_loss
    print("TRAIN: LOGGING LOSS")
    loss_func.logging_loss("train")
    model.logging_unet_image(logging_wandb, epoch)

    print(f"TRAIN: Epoch [{epoch+1}], Batch [{i+1}/{len_train_loader}], Loss: {running_loss}")
    
    # return results
    return model_loss    

def val_with_YoloLoss(model, i, epoch, batch, loss_func, len_val_loader, logging_wandb, is_last=False):
    running_loss = 0.0
    # Чтение данных
    batch["images"] = normalize_pad(batch['images'].float().squeeze(1).to(device))
    images = batch['images'].requires_grad_(True)
    boxes = batch['boxes'] 
    classes = batch['clss']

    # Выпрямление боксов и классов в один вектор
    flat_boxes = torch.stack([box for sublist in boxes for box in sublist]).float()
    flat_classes = torch.tensor([clss for sublist in classes for clss in sublist]).float()
    
    # Генераация batch_idx
    batch_idx = []
    for j, sublist in enumerate(boxes):
        batch_idx.extend(torch.tensor([j] * len(sublist)))
    batch_idx = torch.tensor(batch_idx).float()

    # Вывод модели
    output = model(images)

    # Перегруппировка таргетов для правильного формата
    target = {"batch_idx" : batch_idx, "cls" : flat_classes, "bboxes" : flat_boxes}

    # Рассчет ошибки
    model_loss = loss_func(output, target)
    
    
    #Преобразования выхода в читаемый формат
    # results = preprocess_data(output, target)

    # Усреднение ошибки
    model_loss = model_loss.mean()
    print("My model loss: ", model_loss)
    running_loss = model_loss.item()
    
    if is_last:
        loss_func.result /= len_val_loader
        loss_func.logging_loss("val")
        model.logging_unet_image(logging_wandb, epoch)

        print(f"VAL: Epoch [{epoch+1}], Batch [{i+1}/{len_val_loader}], Loss: {running_loss}")
    
    # return results
    return None

def training_with_ComplexLoss(model, i, epoch, batch, loss_func, len_train_loader, logging_wandb, is_last=False, last_loss=None):
    running_loss = 0.0
    # Чтение данных
    batch["images"] = normalize_pad(batch['images'].float().squeeze(1).to(device))
    print("Image data: ", "Min: ", torch.min(batch["images"]), "Max: ", torch.max(batch["images"]))
    images = batch['images'].requires_grad_(True)
    boxes = batch['boxes'] 
    classes = batch['clss']

    # Выпрямление боксов и классов в один вектор
    flat_boxes = torch.stack([box for sublist in boxes for box in sublist]).float().requires_grad_(True)
    flat_classes = torch.tensor([clss for sublist in classes for clss in sublist]).float().requires_grad_(True)
    print("Box data: ", "Min: ", torch.min(flat_boxes), "Max: ", torch.max(flat_boxes))
    print("Class data: ", "Min: ", torch.min(flat_classes), "Max: ", torch.max(flat_classes))
    
    # Генераация batch_idx
    batch_idx = []
    for j, sublist in enumerate(boxes):
        batch_idx.extend(torch.tensor([j] * len(sublist)))
    batch_idx = torch.tensor(batch_idx).float().requires_grad_(True)

    # Генерация вывода обертки модели (дял проверки)
    yolo = YOLO("yolov8x-oiv7.pt")
    yolo(images[0])

    # Вывод модели
    output = model(images)

    # Перегруппировка таргетов для правильного формата
    target = {"batch_idx" : batch_idx, "cls" : flat_classes, "bboxes" : flat_boxes}

    # Рассчет ошибки
    model_loss = loss_func(model.intermadiate_result, images[:, :3, :, :], output, target)
    
    
    #Преобразования выхода в читаемый формат
    # results = preprocess_data(output, target)

    # Усреднение ошибки
    print("My model loss: ", model_loss)
    running_loss += model_loss.item()
    
    #Обработка ошибки при обучении
    optimizer.zero_grad()
    model_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.yolo.parameters(), max_norm=1.)
    model.check_grads(logging_wandb, running_loss)
    model.logging_grads_info(logging_wandb)
    optimizer.step()
    
    
    if is_last:
        loss_func.result += last_loss
        loss_func.logging_loss("train")
        model.logging_unet_image(logging_wandb, epoch)

    print(f"TRAIN: Epoch [{epoch+1}], Batch [{i+1}/{len_train_loader}], Loss: {running_loss}")
    
    # return results
    return model_loss

def val_with_ComplexLoss(model, i, epoch, batch, loss_func, len_val_loader, is_last=False):
    running_loss = 0.0
    # Чтение данных
    batch["images"] = normalize_pad(batch['images'].float().squeeze(1).to(device))
    print("Image data: ", "Min: ", torch.min(batch["images"]), "Max: ", torch.max(batch["images"]))
    images = batch['images'].requires_grad_(True)
    boxes = batch['boxes'] 
    classes = batch['clss']

    # Выпрямление боксов и классов в один вектор
    flat_boxes = torch.stack([box for sublist in boxes for box in sublist]).float()
    flat_classes = torch.tensor([clss for sublist in classes for clss in sublist]).float()
    print("Box data: ", "Min: ", torch.min(flat_boxes), "Max: ", torch.max(flat_boxes))
    print("Class data: ", "Min: ", torch.min(flat_classes), "Max: ", torch.max(flat_classes))
    
    # Генераация batch_idx
    batch_idx = []
    for j, sublist in enumerate(boxes):
        batch_idx.extend(torch.tensor([j] * len(sublist)))
    batch_idx = torch.tensor(batch_idx).float()

    # Генерация вывода обертки модели (дял проверки)
    yolo = YOLO("yolov8x-oiv7.pt")
    yolo(images[0])

    # Вывод модели
    output = model(images)

    # Перегруппировка таргетов для правильного формата
    target = {"batch_idx" : batch_idx, "cls" : flat_classes, "bboxes" : flat_boxes}

    # Рассчет ошибки
    model_loss = loss_func(model.intermadiate_result, images[:, :3, :, :], output, target)
    
    
    #Преобразования выхода в читаемый формат
    # results = preprocess_data(output, target)

    # Усреднение ошибки
    print("My model loss: ", model_loss)
    running_loss = model_loss.item()
    
    if is_last:
        loss.func.result /= len_val_loader
        loss_func.logging_loss("val")
        model.logging_unet_image(logging_wandb, epoch)

        print(f"VAL: Epoch [{epoch+1}], Batch [{i+1}/{len_val_loader}], Loss: {running_loss}")
    
    # return results
    return None
    
def training_only_FOX(model, i, epoch, batch, loss_func, len_train_loader, logging_wandb, is_last=False, last_loss=None):
    running_loss = 0.0
    # Чтение данных
    batch["images"] = normalize_pad(batch['images'].float().squeeze(1).to(device))
    print("Image data: ", "Min: ", torch.min(batch["images"]), "Max: ", torch.max(batch["images"]))
    images = batch['images'].requires_grad_(True)
    boxes = batch['boxes'] 
    classes = batch['clss']

    # Выпрямление боксов и классов в один вектор
    flat_boxes = torch.stack([box for sublist in boxes for box in sublist]).float().requires_grad_(True)
    flat_classes = torch.tensor([clss for sublist in classes for clss in sublist]).float().requires_grad_(True)
    print("Box data: ", "Min: ", torch.min(flat_boxes), "Max: ", torch.max(flat_boxes))
    print("Class data: ", "Min: ", torch.min(flat_classes), "Max: ", torch.max(flat_classes))
    
    # Генераация batch_idx
    batch_idx = []
    for j, sublist in enumerate(boxes):
        batch_idx.extend(torch.tensor([j] * len(sublist)))
    batch_idx = torch.tensor(batch_idx).float().requires_grad_(True)


    # Вывод модели
    output = model(images[:, :, :, :])

    # Перегруппировка таргетов для правильного формата
    target = images[:, :3, :, :]

    # Рассчет ошибки
    model_loss = loss_func(output, target)

    #Преобразования выхода в читаемый формат
    # results = preprocess_data(output, target)

    print("My model loss: ", model_loss)
    running_loss += model_loss.item()

    #Обработка ошибки при обучении
    optimizer.zero_grad()
    model_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.yolo.parameters(), max_norm=1.)
    model.check_grads(logging_wandb, running_loss)
    model.logging_grads_info(logging_wandb)
    optimizer.step()
    
    
    # if is_last:
    #     loss_func.result += last_loss
    loss_func.logging_loss("train")
    model.logging_unet_image(logging_wandb, epoch)

    print(f"TRAIN: Epoch [{epoch+1}], Batch [{i+1}/{len_train_loader}], Loss: {running_loss}")
    
    # return results
    return model_loss

def val_only_FOX(model, i, epoch, batch, loss_func, len_val_loader, logging_wandb, is_last=False):
    running_loss = 0.0
    # Чтение данных
    batch["images"] = normalize_pad(batch['images'].float().squeeze(1).to(device))
    print("Image data: ", "Min: ", torch.min(batch["images"]), "Max: ", torch.max(batch["images"]))
    images = batch['images']
    boxes = batch['boxes'] 
    classes = batch['clss']

    # Выпрямление боксов и классов в один вектор
    flat_boxes = torch.stack([box for sublist in boxes for box in sublist]).float()
    flat_classes = torch.tensor([clss for sublist in classes for clss in sublist]).float()
    print("Box data: ", "Min: ", torch.min(flat_boxes), "Max: ", torch.max(flat_boxes))
    print("Class data: ", "Min: ", torch.min(flat_classes), "Max: ", torch.max(flat_classes))
    
    # Генераация batch_idx
    batch_idx = []
    for j, sublist in enumerate(boxes):
        batch_idx.extend(torch.tensor([j] * len(sublist)))
    batch_idx = torch.tensor(batch_idx).float()

    # Генерация вывода обертки модели (дял проверки)

    # Вывод модели
    output = model(images[:, :, :, :])

    # Перегруппировка таргетов для правильного формата
    target = images[:, :3, :, :]

    # Рассчет ошибки
    model_loss = loss_func(output, target)

    #Преобразования выхода в читаемый формат
    # results = preprocess_data(output, target)

    # Усреднение ошибки
    print("My model loss: ", model_loss)
    running_loss = model_loss.item()
    
    if is_last:
        loss_func.result /= len_val_loader
        loss_func.logging_loss("val")

        print(f"VAL: Epoch [{epoch+1}], Batch [{i+1}/{len_val_loader}], Loss: {running_loss/len_val_loader}")
    
    # return results
    return None    

