def evaluate_on_bench(loader, uclasses, model, preprocess_data, metric):
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
        if ((i + 1) % 2 == 0) or (i == len(loader)-1):
            # Calculate mean metrics across all classes
            mean_ap = sum([metric_sum[class_id]["ap"] for class_id in uclasses]) / len(uclasses)
            mean_precision = sum([metric_sum[class_id]["precision"] for class_id in uclasses]) / len(uclasses)
            mean_recall = sum([metric_sum[class_id]["recall"] for class_id in uclasses]) / len(uclasses)
            print(f'Iteration [{i+1}/{len(loader)}]:')
            print(f'  Mean AP: {mean_ap/i:.4f}, Mean Precision: {mean_precision/i:.4f}, Mean Recall: {mean_recall/i:.4f}')
        wandb.log({"Mean AP" : mean_ap/i, "Mean Precision" : mean_precision/i, "Mean Recall" : mean_recall/i})

