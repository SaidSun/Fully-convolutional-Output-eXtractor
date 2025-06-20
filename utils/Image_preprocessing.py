# Нормализаци значений пикселей
def normalize_pad(pad_tensor):
    pad_tensor = pad_tensor.float()
    pad_tensor = pad_tensor.clamp(0, 255)
    pad_tensor = pad_tensor / 255.0
    return pad_tensor

# Перевод Numpy в image формат для визуализации
def numpy_to_image(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    print(tensor.shape)
    img = Image.fromarray(tensor)

    return img

# Перевод Tensor в image формат для визуализации
def tensor_to_image(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    tensor = tensor.permute(2, 0, 1)
   
    tensor = (tensor * 255).clamp(0, 255).byte()
    print(tensor[:, 0, 0])
    img_np = tensor.cpu().numpy()
    print(img_np.shape)
    img = Image.fromarray(img_np)

    return img

# Перевод JPEG в Tensor формат для визуализации
def jpeg_to_tensor(self, image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([transforms.ToTensor()])
        tensor_image = transform(image)
        batch_tensor = tensor_image.unsqueeze(0)
        return batch_tensor

# Интерполяция Тензора до указанногго размера target_size
def interpolate_to_fixed_size(images, target_size=(1000, 1000)):
    batch_size, channels, _, _ = images.shape  
    interpolated_images = []
    for i in range(batch_size):
        image = images[i, :, :, :]  # (C, H, W)
        image = image.unsqueeze(0)  # (1, C, H, W)
        interpolated_image = nn.functional.interpolate(image.float(), size=target_size, mode='bilinear', align_corners=False)  # (1, C, 1000, 1000)
        interpolated_images.append(interpolated_image)
    interpolated_images = torch.cat(interpolated_images, dim=0)  # (B, C, 1000, 1000)
    return interpolated_images

# Создание  маски для изображений
def create_channel_masks(images, batch_size, height, width, device="cpu"):
    images_masked = images.clone()
    mask1 = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32, device=device)
    mask2 = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32, device=device)
    rgb_channels = []
    for i in range(batch_size):
        channels = random.sample([0, 1, 2], 2)
        images_masked[i, channels[0], :, :] = images[i, channels[0], :, :] * mask1[i, :, :, :]
        images_masked[i, channels[1], :, :] = images[i, channels[1], :, :] * mask2[i, :, :, :]
    return images_masked

# Разбитие исходного изображения на непересекающиеся патчи
def create_patches(images, patch_size = (100, 100), stride=None, drop_last=False):
    batch_size, channels, height, width = images.shape
    if isinstance(patch_size, int):
        patch_height = patch_size
        patch_width = patch_size
    else:
        patch_height, patch_width = patch_size

    if stride is None:
        stride_height = patch_height
        stride_width = patch_width
    elif isinstance(stride, int):
        stride_height = stride
        stride_width = stride
    else:
        stride_height, stride_width = stride
    num_patches_h = (height - patch_height) // stride_height + 1
    num_patches_w = (width - patch_width) // stride_width + 1
    pad_h = 0 if drop_last else max(0, (stride_height * (num_patches_h - 1) + patch_height - height))
    pad_w = 0 if drop_last else max(0, (stride_width * (num_patches_w - 1) + patch_width - width))
    if pad_h > 0 or pad_w > 0:
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        images = nn.functional.pad(images, padding, mode='reflect')
        _, _, height, width = images.shape

   
        num_patches_h = (height - patch_height) // stride_height + 1
        num_patches_w = (width - patch_width) // stride_width + 1


 
    patches = images.unfold(2, patch_height, stride_height).unfold(3, patch_width, stride_width)


    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()


    patches = patches.view(-1, channels, patch_height, patch_width)

    return patches, num_patches_h, num_patches_w

# Восстановление непересекающихся патчей в полноценное изображение
def patches_to_images(model_image, image, num_patches_h, num_patches_w):
    batch_size, _, height, width = image.shape
    patch_image_height = int(num_patches_h * model_image.shape[2])
    patch_image_width = int(num_patches_w * model_image.shape[3])
    pad_h = patch_image_height - height
    pad_w = patch_image_width - width
    model_image = model_image.view(batch_size, 3, patch_image_height, patch_image_width)
    return model_image, nn.functional.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='reflect')

# Часть Софии
# Глубина модели Depth Anything (абсолютная глубина)
def depthDP(image):
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        outputs = model_depthDP(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
      outputs,
      target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]

    min = predicted_depth.min().item()
    max = predicted_depth.max().item()

    max_depth = 100
    normalized_depth = predicted_depth.detach().cpu().numpy() / max_depth
    normalized_depth = Image.fromarray((normalized_depth * 255).astype("uint8"))
    return normalized_depth.resize(image_size)

# Глубина модели Depth Anything (относительная глубина)
def depthDA(image):
    depth = model_depthDA(image)["depth"]
    depth_map = np.array(depth)

    normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    normalized_depth_map = normalized_depth_map.astype(np.uint8)

    depth_image = Image.fromarray(normalized_depth_map)

    return np.asarray(depth_image.resize(image_size))

# Генерация 5-ти канальных изображнеий
def five_channels(image, depth_kind="DA"):
    image = transforms.functional.to_pil_image(image)
    
    if depth_kind == "DA":
        image_depth = np.expand_dims(depthDA(image), axis=2)
    else:
        image_depth = np.expand_dims(depthDP(image), axis=2)

    image_edge = np.expand_dims(cv2.Canny(np.array(image), t_lower, t_upper), axis=2)

    image_5channels = torch.permute(torch.from_numpy(np.concatenate((image, image_depth, image_edge), 
                                                                    axis=2)), (2, 0, 1))
    image_5channels = image_5channels.unsqueeze(0)

    return image_5channels

