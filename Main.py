# Selecting a processing unit


dtype = torch.float16

model_depthDA = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

Checkpoint = "apple/DepthPro-hf"
image_processor = AutoImageProcessor.from_pretrained(Checkpoint, use_fast=True)
model_depthDP =  AutoModelForDepthEstimation.from_pretrained(
    Checkpoint, device_map=device, torch_dtype=dtype, use_fov_model=True
)


# Choose multiple units

if device == "cuda":
    # USE MULTIPLE GPUS
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")
else:
    print("\\Error\\")


# Logging Wandb

wandb.login(key="your_key_here")


## Merging with YOLO

# Creating training and test datasets
print("Start")
data = torch.load(path_train_dataset)
convert_classes_to_yolo(data, convert_classes)
flat_classes = torch.tensor([clss for sublist in data["cls"] for clss in sublist]).float().requires_grad_(True)
uclasses = list(map(int, torch.unique(torch.tensor(flat_classes))))
print("End")

_, data = split_data(data)
# train_data, test_data = split_data(data)
train_data, test_data = split_data(data)
train_dataset = TensorDataset(train_data, depth_kind=depth_kind)
validation_dataset = TensorDataset(test_data, depth_kind=depth_kind)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Init dual-component model

dual_model = TwoStepModel(levels, channels, lvls_kernel, pools, device=device)
dual_model = dual_model.to(device)
yolo_model = YOLOv8model(device=device)
fox_model = FOXModel(levels, channels, lvls_kernel, pools, use_checkpoint=True, device = device)


# Init loss function

loss_func = YOLOv8Loss(dual_model.yolo).to(device)
loss_func = FOXLoss(mse_weight=0.8)


# Init optimizer

# opt_params = [
#     {'params': dual_model.u_net.get_color_params(), 'lr': 1e-4},  # уменьшено
#     {'params': dual_model.u_net.get_depth_params(), 'lr': 1e-5},   # уменьшено
#     {'params': dual_model.u_net.get_contour_params(), 'lr': 1e-4}, # уменьшено
#     {'params': dual_model.u_net.get_shared_params(), 'lr': 1e-3}, # уменьшено
#     {'params': dual_model.yolo.parameters(), 'lr': 1e-5},          # значительно уменьшеноэ
#     # {"params" : loss_func.RGBTargetLoss.weights.parameters(), "lr" : 1e-3}
# ]
opt_params = [
    {'params': fox_model.u_net.get_color_params(), 'lr': 1e-4},  # уменьшено
    {'params': fox_model.u_net.get_depth_params(), 'lr': 1e-5},   # уменьшено
    {'params': fox_model.u_net.get_contour_params(), 'lr': 1e-4}, # уменьшено
    {'params': fox_model.u_net.get_shared_params(), 'lr': 1e-3}, # уменьшено
    # {"params" : loss_func.RGBTargetLoss.weights.parameters(), "lr" : 1e-3}
]
optimizer = optim.Adam(opt_params)


# Transferring the model to training mode

dual_model.train()


# Init the wandb logger
my_wandb = Wandb_unit()
conf = {"learning_rate": "1e-5, 1e-6, 1e-6, 1e-5, 1e-4",
        "architecture": "FOXv3",
        "dataset": 1000,
        "epochs": 5,
        "batch_size": 2,
        "criterion": f"YOLOv8Loss"  #RGBTargetAwareLoss() weight=0.3, 0.3, 0.3 #f"Complex loss, weights [0.1, 1.], RGBTargWeights [0.3, 0.3, 0.3]", 
            }
proj = "saidthefoxsun-mai"


# Training and testing dualmodel together

train_val_model(fox_model, num_epochs, train_loader, validation_loader, my_wandb, conf, proj, loss_func, training_only_FOX, val_only_FOX)

my_wandb.fin()


torch.save(dual_model.state_dict(), "YOLO+FOX_DA_v4.pth")


## Testing on benchmark

# Init model

model = YOLOv8model(device = device)

model = TwoStepModel(levels, channels, lvls_kernel, pools, use_checkpoint=True, model_path="yolov8x-oiv7.pt", device = device)

model.load_state_dict(torch.load("/home/jupyter/datasphere/project/YOLO+FOX_DA_v3.pth"))
model.to(device)


# Reading benchmark data

print("Start")
bench_data = torch.load(path_to_bench_data)
convert_classes_to_yolo(bench_data, convert_classes_bench)
dataset = TensorDataset(bench_data)
loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
flat_classes = torch.tensor([clss for sublist in bench_data["cls"] for clss in sublist]).float().requires_grad_(True)
uclasses = list(map(int, torch.unique(torch.tensor(flat_classes))))
print("End")


# Init preprocess data

preprocess_data = preprocess_yolo_data(model.yolo)

dataset = TensorDataset(bench_data, depth_kind=depth_kind)
loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)


# Init metric

metric = PnRn_mAP(uclasses = uclasses, iou_thresholds=[0.5])


## Make evaluation

metric_sum = {class_id: {"ap": 0.0, "precision": 0.0, "recall": 0.0} for class_id in uclasses}
for i, batch in enumerate(loader):
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

    target = {"batch_idx" : batch_idx, "cls" : flat_classes, "bboxes" : flat_boxes}
    
    # Вывод модели
    output = model(images[:, :3, :, :])
    results = preprocess_data(output, target)
    metric_batch = metric(results["pred_bboxes"], results["pred_scores"], results["target_bboxes"], results["fg_mask"], results["target_labels"])[0.5]
    for class_id in uclasses:
        metric_sum[class_id]["ap"] += metric_batch[class_id]["ap"]
        metric_sum[class_id]["precision"] += metric_batch[class_id]["precision"]
        metric_sum[class_id]["recall"] += metric_batch[class_id]["recall"]
    if (i + 1) % 2 == 0:
        # Calculate mean metrics across all classes
        mean_ap = sum([metric_sum[class_id]["ap"] for class_id in uclasses]) / len(uclasses)
        mean_precision = sum([metric_sum[class_id]["precision"] for class_id in uclasses]) / len(uclasses)
        mean_recall = sum([metric_sum[class_id]["recall"] for class_id in uclasses]) / len(uclasses)
        print(f'Iteration [{i+1}/{len(loader)}]:')
        print(f'  Mean AP: {mean_ap/i:.4f}, Mean Precision: {mean_precision/i:.4f}, Mean Recall: {mean_recall/i:.4f}')


mean_ap = sum([metric_sum[class_id]["ap"] for class_id in uclasses]) / len(uclasses)
mean_precision = sum([metric_sum[class_id]["precision"] for class_id in uclasses]) / len(uclasses)
mean_recall = sum([metric_sum[class_id]["recall"] for class_id in uclasses]) / len(uclasses)
print(metric_batch)
print(f'Iteration [{i+1}/{len(loader)}]:')
print(f'  Mean AP: {mean_ap/i:.4f}, Mean Precision: {mean_precision/i:.4f}, Mean Recall: {mean_recall/i:.4f}')


torch.save(dual_model.state_dict(), "YOLO+FOX_on")


torch.cuda.empty_cache()

my_wandb.fin()

