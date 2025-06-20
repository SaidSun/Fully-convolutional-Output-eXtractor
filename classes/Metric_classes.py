# Metric classes 

class PSNR:
    def __init__(self):
        pass
    
    def calculate_average_psnr_linear(self, psnr_values):
        """Вычисляет средний PSNR, преобразуя в линейную шкалу."""
        mse_values = [10**(-psnr / 10) for psnr in psnr_values] # Преобразование PSNR в MSE
        average_mse = sum(mse_values) / len(mse_values) # Вычисляем среднее MSE
        average_psnr = -10 * np.log10(average_mse) # Преобразуем обратно в PSNR (dB)
        return average_psnr
    
    def __call__(self, img1, img2, data_range=[-304, 255]):
        """
        Вычисляет Peak Signal-to-Noise Ratio (PSNR) между двумя изображениями.

        Args:
            img1 (torch.Tensor or numpy.ndarray): Первое изображение.
            img2 (torch.Tensor or numpy.ndarray): Второе изображение.
            data_range (float): Максимальный возможный диапазон значений пикселей.
                                 Используйте 1.0 для изображений в диапазоне [0, 1],
                                 и 255.0 для изображений в диапазоне [0, 255].

        Returns:
            float: Значение PSNR.
        """

        # Преобразуем в numpy, если входные данные - тензоры PyTorch
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()  # отсоединяем от графа и переносим на CPU
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()

        # Преобразуем к float64 для стабильности расчетов
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # Вычисляем PSNR с помощью skimage.metrics.peak_signal_noise_ratio
        psnr_value = peak_signal_noise_ratio(img1, img2, data_range=data_range)

        return self.calculate_average_psnr_linear(psnr_value)
    
class FSIM:
    def __init__(self):
        pass
    
    def gradient(self, img):
        dx = np.gradient(img, axis=1) # градиент по x
        dy = np.gradient(img, axis=0) # градиент по y
        return np.sqrt(dx**2 + dy**2)
    
    def phase_cong(img):
        # Это упрощенная версия, в реальной реализации PC сложнее
        return np.abs(img)
    
    def __call__(self, img1, img2):
        """
        Вычисляет Feature Similarity Index (FSIM) между двумя изображениями.
        Основан на статье:
        Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). FSIM: A feature similarity index for image quality assessment. IEEE Transactions on Image Processing, 20(8), 2378-2386.
        """
        # 1. Преобразование в numpy массивы и float
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()

        # 2. Проверка, что изображения в формате grayscale
        if img1.ndim == 3:
            img1 = np.mean(img1, axis=2) # Преобразуем RGB в grayscale
        if img2.ndim == 3:
            img2 = np.mean(img2, axis=2)

        # 3. Параметры
        T1 = 0.85
        T2 = 160
        alpha = 1
        beta = 1

        # 4. Вычисление градиента
    
        Gm1 = self.gradient(img1)
        Gm2 = self.gradient(img2)

        # 5. Вычисление фазовой конгруэнтности (Phase Congruency - PC)
        
        PC1 = self.phase_cong(img1)
        PC2 = self.phase_cong(img2)

        # 6. Вычисление Similarity Maps
        S_G = (2 * Gm1 * Gm2 + T1) / (Gm1**2 + Gm2**2 + T1)
        S_PC = (2 * PC1 * PC2 + T2) / (PC1**2 + PC2**2 + T2)

        # 7. Вычисление FSIM
        fsim_map = (S_G**alpha * S_PC**beta)
        fsim_value = np.mean(fsim_map)

        return fsim_value
    
class PnRn_mAP(nn.Module):
    def __init__(self, uclasses=[range(0, 602)], iou_thresholds=[0.5]):
        super().__init__()
        self.uclasses = uclasses
        self.iou_thresholds = iou_thresholds

    def forward(self, pred_bboxes, pred_scores, target_bboxes, fg_mask, target_labels):
        map_values = {}
        for iou_threshold in self.iou_thresholds:
            map_values[iou_threshold] = self.calculate_map(
                pred_bboxes, pred_scores, target_bboxes, 
                fg_mask, target_labels, iou_threshold
            )
        return map_values

    def calculate_map(self, pred_bboxes, pred_scores, target_bboxes, fg_mask, target_labels, iou_threshold):
        batch_size = target_bboxes.shape[0]
        map_values = {class_id: {"ap": 0.0, "precision": 0.0, "recall": 0.0} for class_id in self.uclasses}

        for b in range(batch_size):
            # Применяем маску к предсказаниям и целям
            pred_bboxes_fg = pred_bboxes[b][fg_mask[b]]
            pred_scores_fg = pred_scores[b][fg_mask[b]]
            target_bboxes_fg = target_bboxes[b][fg_mask[b]]
            target_labels_fg = target_labels[b][fg_mask[b]]

            if len(pred_bboxes_fg) == 0:
                continue

            # 1. Определяем предсказанные классы
            pred_classes = pred_scores_fg.argmax(dim=1)
            print("PRED_CLASSES: ", pred_classes)
            print("TARGETS: ", target_labels_fg)
            # 2. Вычисляем попарное IoU между соответствующими объектами
            ious = self.pairwise_bbox_iou(pred_bboxes_fg, target_bboxes_fg)
            
            # 3. Для каждого класса считаем метрики
            for class_id in self.uclasses:
                # Фильтруем предсказания и цели класса
                class_mask = (pred_classes == class_id) & (target_labels_fg == class_id)
                class_tp = (ious[class_mask] >= iou_threshold).float().sum()
                
                # Общее количество объектов класса
                total_pred = (pred_classes == class_id).sum().float()
                total_target = (target_labels[b] == class_id).sum().float()  # Все цели в батче

                # Расчет precision и recall
                precision = class_tp / (total_pred + 1e-6)
                recall = class_tp / (total_target + 1e-6)
                
                # AP как площадь под кривой (упрощенно)
                ap = precision * recall
                # print(f"Score metrics:\nAP: {ap}\nPrecision: {precision}\nRecall: {recall}")
                # Накопление значений
                map_values[class_id]["ap"] += ap.item()
                map_values[class_id]["precision"] += precision.item()
                map_values[class_id]["recall"] += recall.item()

        # Усреднение по батчу
        for class_id in self.uclasses:
            map_values[class_id]["ap"] /= batch_size
            map_values[class_id]["precision"] /= batch_size
            map_values[class_id]["recall"] /= batch_size

        return map_values

    def pairwise_bbox_iou(self, box1, box2):
        """Вычисляет IoU для каждой пары соответствующих боксов"""
        lt = torch.max(box1[:, :2], box2[:, :2])
        rb = torch.min(box1[:, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union = area1 + area2 - inter
        return inter / (union + 1e-6)

