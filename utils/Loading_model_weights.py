def load_weights(model):
    try:
        state_dict = torch.load(weights, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Файл не найден.  Убедитесь, что он находится в правильном месте.")
    return model


# Split data

def split_data(data, train_ratio=0.8):

    dataset_size = len(data["cls"])  # Используем длину классов как размер датасета
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    print("Size: ", dataset_size)
    # Создаем индексы для разделения
    train_indices, test_indices = torch.randperm(dataset_size).split([train_size, test_size])
    
    # Создаем тренировочный датасет
    train_data = {
        "images": data["images"][train_indices],  # Индексируем изображения
        "batch_idx": [data["batch_idx"][i] for i in train_indices],  # Индексируем батч индексы
        "cls": [data["cls"][i] for i in train_indices],  # Индексируем метки классов
        "bboxes": [data["bboxes"][i] for i in train_indices],  # Индексируем bounding boxes
    }

    # Создаем тестовый датасет
    test_data = {
        "images": data["images"][test_indices],  # Индексируем изображения
        "batch_idx": [data["batch_idx"][i] for i in test_indices],  # Индексируем батч индексы
        "cls": [data["cls"][i] for i in test_indices],  # Индексируем метки классов
        "bboxes": [data["bboxes"][i] for i in test_indices],  # Индексируем bounding boxes
    }

    return train_data, test_data


# Using model on benchmark

def using_on_benchmark(model, weights="/home/jupyter/datasphere/project/5_channel_new_model.pth",
                       data_path="/home/jupyter/datasphere/project/dataset_benchmark.pth",
                       folder_path="/home/jupyter/datasphere/project/orig_images/"):
    data_torch = torch.load(data_path)
    if not(weights in None):
        model = load_weights(model)
    model = model.to(device)
    imagess = []
    for k, i in enumerate(data_torch):
        filename = f"{folder_path}image{k+1}.jpg"
        w, h = get_jpeg_size_pil(filename)
        model_img = model(normalize_pad(i.to(device)))[1]
        model_img = nn.functional.interpolate(model_img, size=(h, w))
        imagess.append(model_img)
    torch.save(imagess, "generated_images.pth")
    print("Finish")


# Converting classes

def create_reversed_dictionary(dictionary):
    reversed_dict = {value: key for key, value in dictionary.items()}
    return reversed_dict

def convert_classes_to_yolo(data, convert_classes):
    if "cls" not in data:
        print("Предупреждение: Ключ 'cls' отсутствует в данных.")
        return data

    new_cls = []
    for image_classes in data["cls"]:  # data["cls"] - это список списков классов для каждого изображения
        new_image_classes = []
        for old_class_id in image_classes:
            print(int(old_class_id))
            if int(old_class_id) in convert_classes.keys():
                new_class_id = convert_classes[int(old_class_id)]
                new_image_classes.append(new_class_id)
            else:
                print(f"Предупреждение: Старый номер класса {old_class_id} отсутствует в словаре convert_classes.  Пропускаем этот класс.")
                #Если старого класса нет в словаре преобразований, можно:
                # 1. Пропустить этот класс (как сделано здесь)
                # 2. Присвоить ему класс "unknown" (если такой класс есть в YOLO)
                # 3. Поднять исключение (если отсутствие класса - ошибка)
        new_cls.append(new_image_classes)  # Добавляем сконвертированный список классов для изображения в общий список
    data["cls"] = new_cls
    return data


model = torch.load('/home/jupyter/datasphere/project/yolov8x-oiv7.pt')  # Загрузка модели
class_names =create_reversed_dictionary(model["model"].names)  # Получаем словарь классов
convert_classes = {0: class_names["Car"],
  1: class_names["Clothing"],
  2: class_names["Van"],
  3: class_names["Bicycle"],
  4: class_names["Car"],
  5: class_names["Wheel"],
  6: class_names["Billboard"],
  7: class_names["Bus"],
  8: class_names["Tire"],
  9: class_names["Traffic light"],
  10: class_names["Bicycle wheel"],
  11: class_names["Boat"],
  12: class_names["Land vehicle"],
  13: class_names["Tree"],
  14: class_names["Footwear"],
  15: class_names["Jeans"],
  16: class_names["Tower"],
  17: class_names["Window"],
  18: class_names["House"],
  19: class_names["Vehicle registration plate"],
  20: class_names["Man"],
  21: class_names["Motorcycle"],
  22: class_names["Taxi"],
  23: class_names["Street light"],
  24: class_names["Chair"],
  25: class_names["Woman"],
  26: class_names["Person"],
  27: class_names["Truck"],
  28: class_names["Traffic sign"],
  29: class_names["Flower"],
  30: class_names["Sculpture"],
  31: class_names["Sculpture"]}

convert_classes_bench = {0: class_names["Bicycle wheel"],
  1: class_names["Bicycle"],
  2: class_names["Billboard"],
  3: class_names["Boat"],
  4: class_names["Bus"],
  5: class_names["Car"],
  6: class_names["Chair"],
  7: class_names["Clothing"],
  8: class_names["Flower"],
  9: class_names["Footwear"],
  10: class_names["House"],
  11: class_names["Jeans"],
  12: class_names["Land vehicle"],
  13: class_names["Man"],
  14: class_names["Motorcycle"],
  15: class_names["Sculpture"],
  16: class_names["Person"],
  17: class_names["Sculpture"],
  18: class_names["Street light"],
  19: class_names["Taxi"],
  20: class_names["Tire"],
  21: class_names["Tower"],
  22: class_names["Traffic light"],
  23: class_names["Traffic sign"],
  24: class_names["Tree"],
  25: class_names["Truck"],
  26: class_names["Van"],
  27: class_names["Vehicle registration plate"],
  28: class_names["Wheel"],
  29: class_names["Window"],
  30: class_names["Woman"],
  31: class_names["Vehicle"]}

# Prepocess YOLO output data

class preprocess_yolo_data:
    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        m = model.model[-1]
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.assigner = TaskAlignedAssigner(topk = tal_topk, num_classes = self.nc, alpha=0.5, beta=0.6)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.use_dfl = m.reg_max > 1

    def preprocess(self, targets, batch_size, scale_tensor):
       """Preprocess targets by converting to tensor format and scaling coordinates."""
       nl, ne = targets.shape
       if nl == 0:
           out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
       else:
           i = targets[:, 0]  # image index
           _, counts = i.unique(return_counts=True)
           counts = counts.to(dtype=torch.int32)
           out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
           for j in range(batch_size):
               matches = i == j
               if n := matches.sum():
                   out[j, :n] = targets[matches, 1:]
           out_xywh = out*1
           out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
       return out, out_xywh

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False), dist2bbox(pred_dist, anchor_points, xywh=True)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        print("Before: ", targets.shape)
        targets, targets_xywh = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        print("After: ", targets.shape)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        # Pboxes
        pred_bboxes, pred_bboxes_xywh = self.bbox_decode(anchor_points, pred_distri)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        num_total_anchors = pred_scores.shape[1]
        results = {"pred_scores" : pred_scores, 
                   "pred_bboxes" : pred_bboxes, 
                   "pred_bboxes_xywh" : pred_bboxes_xywh/(stride_tensor*10.0), 
                   "target_bboxes" : target_bboxes/stride_tensor, 
                   "target_scores" : target_scores,
                   "target_labels" : target_labels,
                   "target_gt_idx" : target_gt_idx,
                   "fg_mask" : fg_mask}
        # return pred_scores, pred_bboxes, pred_bboxes_xywh/stride_tensor/10.0, gt_labels, gt_bboxes, mask_gt, target_labels, target_bboxes/stride_tensor, target_scores, target_gt_idx, fg_mask, targets_xywh
        return results

