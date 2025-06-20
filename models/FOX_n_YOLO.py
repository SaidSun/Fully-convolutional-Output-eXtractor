# Ultramodel class

class TwoStepModel(nn.Module):
    def __init__(self, levels, channels, lvls_kernel, pools, use_checkpoint=True, model_path="yolov8x-oiv7.pt", device = "cpu"):
        super().__init__()
        self.u_net = UNET(levels, channels, lvls_kernel, pools, use_checkpoint)
        self.yolo = YOLO(model_path).model.to(device)
        self.nc = self.yolo.model[-1].nc  # Количество классов
        self.stride = self.yolo.stride    # Стрaйд для анкеров
        
    def forward(self, X):
        self.intermadiate_result = nn.functional.interpolate(self.u_net(X), size=(640, 640)).requires_grad_(True)
        self.intermadiate_result = self.intermadiate_result.clamp(0, 1).requires_grad_(True)
        # print(f"Max val: {torch.max(self.intermadiate_result)}\nMin val: {torch.min(self.intermadiate_result)}")
        print(self.intermadiate_result.requires_grad)
        return self.yolo(self.intermadiate_result)
    
    def logging_unet_image(self, logging_wandb, epoch):
        imagess = []
        # логгирование как внутренний метод биг модел
        ims = wandb.Image(self.intermadiate_result[1, 0, :, :], caption=f"Epoch: {epoch}")
        imagess.append(ims)
        ims = wandb.Image(self.intermadiate_result[1, 1, :, :], caption=f"Epoch: {epoch}")
        imagess.append(ims)
        ims = wandb.Image(self.intermadiate_result[1, 2, :, :], caption=f"Epoch: {epoch}")
        imagess.append(ims)
        ims = wandb.Image(self.intermadiate_result[1, :, :, :], caption=f"Epoch: {epoch}")
        imagess.append(ims)
        print(self.intermadiate_result.requires_grad)
        wandb.log({"intermediate result" : imagess })
        
    def check_grads(self, logging_wandb, running_loss):
        self.grad_norms = {}
        for name, param in self.u_net.named_parameters():
            if param.grad is not None:
                self.grad_norms[name] = param.grad.norm().item()
            # else:
            #     print("Что-то не так!!!!")
            #     print(name)
        
        for name, param in self.yolo.named_parameters():
            if param.grad is not None:
                self.grad_norms[name] = param.grad.norm().item()
            # else:
            #     print("!!Что-то не так!!!!")
            #     print(name)
        wandb.log({
                **{f"grad_norm/{name}": norm for name, norm in self.grad_norms.items()},
                "batch_loss": running_loss
            })
                
    def logging_grads_info(self, logging_wandb):
        gradients = []
        names = []
        for name, param in self.u_net.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.flatten())
                names.append(name)
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
    
    def train(self):
        for param in self.yolo.parameters():
            param.requires_grad = True
        for param in self.u_net.parameters():
            param.requires_grad = True
        print(next(self.yolo.parameters()).requires_grad)
        print(next(self.u_net.parameters()).requires_grad)
        
        
    def train_yolo(self):
        for param in self.yolo.parameters():
            param.requires_grad = True
        
        for param in self.u_net.parameters():
            param.requires_grad = False
    def train_u_net(self):
        for param in self.yolo.parameters():
            param.requires_grad = False
        
        for param in self.u_net.parameters():
            param.requires_grad = True

    def test(self):
        for param in self.yolo.parameters():
            param.requires_grad = False
        for param in self.u_net.parameters():
            param.requires_grad = False
    

