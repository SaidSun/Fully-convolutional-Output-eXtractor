# YOLOv8 class

class YOLOv8model(nn.Module):
    def __init__(self, model_path="yolov8x-oiv7.pt", device = "cuda"):
        super().__init__()
        self.yolo = YOLO(model_path).model.to(device)
        self.nc = self.yolo.model[-1].nc  # Количество классов
        self.stride = self.yolo.stride    # Стрaйд для анкеров
        self.check_pretrained_weights() # Добавляем проверку при инициализации

    def check_pretrained_weights(self):
        """Проверяет, что модель содержит предобученные веса."""
        for name, param in self.yolo.named_parameters():
            if param.requires_grad:  # Проверяем только обучаемые параметры
                if torch.all(param == 0):
                    print(f"WARNING: Параметр {name} состоит только из нулей. Возможно, веса не загружены.")
                    return False  # Обнаружены нулевые веса, скорее всего, модель не загружена
                else:
                    print(f"Параметр {name}: min={param.min().item():.4f}, max={param.max().item():.4f}")
        print("Модель содержит предобученные веса (проверка пройдена).")
        return True   # Проверка пройдена, модель загружена
        
    def forward(self, X):
        print(f"Max val: {torch.max(X)}\nMin val: {torch.min(X)}")
        print(X.requires_grad)
        return self.yolo(X)
    
    def train(self):
        for param in self.yolo.parameters():
            param.requires_grad = True
            
    def test(self):
        for param in self.yolo.parameters():
            param.requires_grad = False
            
    def check_grads(self, logging_wandb, running_loss):
        self.grad_norms = {}
            
        for name, param in self.yolo.named_parameters():
            if param.grad is not None:
                self.grad_norms[name] = param.grad.norm().item()

        wandb.log({
                **{f"grad_norm/{name}": norm for name, norm in self.grad_norms.items()},
                "batch_loss": running_loss
            })
                
    def logging_grads_info(self, logging_wandb):
        gradients = []
        names = []
        for name, param in self.yolo.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.flatten())
                names.append(name)

        if gradients:
            all_grads = torch.cat(gradients)
            wandb.log({
                "gradients/histogram": wandb.Histogram(all_grads.cpu().numpy()),
                "gradients/max": all_grads.max().item(),
                "gradients/min": all_grads.min().item(),
                "gradients/mean": all_grads.mean().item(),
                "gradients/std": all_grads.std().item()
            })

