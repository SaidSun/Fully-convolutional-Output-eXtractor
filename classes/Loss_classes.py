# Loss classes

class FOXLoss(nn.Module):
    def __init__(self,  mse_weight=0.0):
        super(FOXLoss, self).__init__()
        self.weights_per_ch = torch.eye(3)
        self.losses = [SSIMLossWithWeights(alpha=self.weights_per_ch[0][0]*5, mse_weight = mse_weight), 
                       SSIMLossWithWeights(beta=self.weights_per_ch[1][1]*5,  mse_weight = mse_weight), 
                       SSIMLossWithWeights(gamma=self.weights_per_ch[2][2]*10,  mse_weight = mse_weight)]
        self.result = 0.0
    
    def forward(self, img1, img2):
        print(img1.size(), torch.stack([img1[:, 0, :, :], img1[:, 0, :, :], img1[:, 0, :, :]], dim=1).size())
        ch1_loss = self.losses[0](torch.stack([img1[:, 0, :, :], img1[:, 0, :, :], img1[:, 0, :, :]], dim=1), img2)
        ch2_loss = self.losses[1](torch.stack([img1[:, 1, :, :], img1[:, 1, :, :], img1[:, 1, :, :]], dim=1), img2)
        ch3_loss = self.losses[2](torch.stack([img1[:, 2, :, :], img1[:, 2, :, :], img1[:, 2, :, :]], dim=1), img2)
        self.result += (-(ch1_loss + ch2_loss + ch3_loss)/3)
        return self.result
    
    def clear_loss(self):
         self.result = 0.0
            
    def logging_loss(self, style="train"):
        wandb.log({f"SSIMLoss_{style}" : self.result})
        

class SSIMLossWithWeights(nn.Module):
    def __init__(self, data_range=1.0, win_size=3, win_sigma=5.5, channel=3, 
                 alpha=0.25, beta=0.25, gamma=0.5, size_average=True, epsilon=1e-5, mse_weight=0.5):  # Added l1_weight
        super(SSIMLossWithWeights, self).__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.channel = channel
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.size_average = size_average
        self.epsilon = epsilon
        self.mse_weight = mse_weight  # Weight for L1 loss
        self.result = 0.0
        self.window = self._create_window(win_size, win_sigma, channel)

    def _create_window(self, win_size, win_sigma, channel):
        def gaussian_kernel(size, sigma):
            x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
            gauss = torch.exp(-(x**2) / (2 * sigma**2))
            kernel = gauss / gauss.sum()
            return kernel

        window_1D = gaussian_kernel(win_size, win_sigma)
        window_2D = torch.outer(window_1D, window_1D)
        window = window_2D.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, win_size, win_size).contiguous()
        return window

    def forward(self, img1, img2):

        height1, width1 = img1.shape[-2:]
        height2, width2 = img2.shape[-2:]

        if (height1, width1) != (height2, width2):
            img2 = nn.functional.interpolate(img2, size=(height1, width1), mode='bilinear', align_corners=False)

        device = img1.device
        window = self.window.to(device)

        C1 = (0.01 * self.data_range)**2
        C2 = (0.03 * self.data_range)**2

        mu1 = nn.functional.conv2d(img1, window, padding=self.win_size // 2, groups=self.channel)
        mu2 = nn.functional.conv2d(img2, window, padding=self.win_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=self.win_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=self.win_size // 2, groups=self.channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=self.win_size // 2, groups=self.channel) - mu1_mu2

        l = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)).clamp(0, 1).pow(self.alpha)
        c = ((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)).clamp(0, 1).pow(self.beta)
        s = ((sigma12 + C2/2) / (torch.sqrt(sigma1_sq * sigma2_sq + self.epsilon) + C2/2)).clamp(0, 1).pow(self.gamma) # Добавили epsilon под корень
        ssim_map = l * c * s

        valid_indices = ~torch.isnan(ssim_map)
        if valid_indices.any():
            mean_val = ssim_map[valid_indices].mean()
            ssim_map = torch.where(torch.isnan(ssim_map), mean_val, ssim_map)
        else:
            ssim_map = torch.nan_to_num(ssim_map, nan=0.0, posinf=0.0, neginf=0.0)

        if self.size_average:
            ssim_loss = 1 - ssim_map.mean()
        else:
            ssim_loss = 1 - ssim_map.mean(dim=[1,2,3])
        mse = nn.MSELoss()
        mse_loss = mse(img1[:, :3, :, :], img2[:, :3, :, :])
        # l1_loss = nn.functional.l1_loss(img1, img2, reduction='mean' if self.size_average else 'none')

        self.result = (1 - self.mse_weight) * ssim_loss - self.mse_weight * mse_loss

        return self.result
    
    def clear_loss(self):
         self.result = 0.0
            
    def logging_loss(self, style="train"):
        wandb.log({f"SSIMLoss_{style}" : self.result})
    

class RGBTargetAwareLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # Инициализация Sobel фильтров
        self._init_sobel(device)
        
        # Индивидуальные веса для каналов
        self.weights = nn.ParameterDict({
            'color': nn.Parameter(torch.tensor(0.3)),
            'depth': nn.Parameter(torch.tensor(0.3)),
            'contour': nn.Parameter(torch.tensor(0.3))
        })
        self.result = 0

    def _init_sobel(self, device):
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=device)
        
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

    def forward(self, output, target):
        # Разделение выходных каналов
        color_pred, depth_pred, contour_pred = torch.split(output, 1, dim=1)
        
        # Целевые преобразования
        color_target = target.mean(dim=1, keepdim=True)
        depth_target = 0.299*target[:,0] + 0.587*target[:,1] + 0.114*target[:,2]
        contour_target = self._sobel_magnitude(color_target)

        # Индивидуальные функции потерь
        color_loss = nn.functional.l1_loss(color_pred, color_target)
        depth_loss = 0.6 * nn.functional.l1_loss(depth_pred, depth_target.unsqueeze(1)) + 0.4 * nn.functional.l1_loss(color_pred, color_target)
        contour_loss = nn.functional.binary_cross_entropy_with_logits(contour_pred, contour_target)
        self.result =+ (self.weights['color'] * color_loss +
                self.weights['depth'] * depth_loss +
                self.weights['contour'] * contour_loss)
        # Взвешенная сумма
        return self.result
    
    def clear_loss(self):
         self.result = 0

    def _sobel_magnitude(self, x):
        grad_x = nn.functional.conv2d(x, self.sobel_x, padding=1)
        grad_y = nn.functional.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6) 
    
    def logging_loss(self, style="train"):
        wandb.log({f"Color weight_{style}" : self.weights['color'].item(), 
                   f"Depth Weight_{style}" : self.weights['depth'].item(), 
                   f"Contour Weight_{style}" : self.weights['contour'].item(), 
                   f"RGBLoss_{style}" : self.result})

class Hyperparams:
    def __init__(self, box, cls, dfl):
        self.box = box
        self.cls = cls
        self.dfl = dfl

class YOLOv8Loss(nn.Module):
    def __init__(self, model, batch_size=5.0):
        super().__init__()
        # Инициализация параметров, которые ожидает v8DetectionLoss
        model.args = Hyperparams(7.5, 0.5, 1.5)
        self.IoU_arg = model.args.box
        self.batch_size = batch_size
        
        # Инициализация оригинальной функции потерь
        self.loss_fn = v8DetectionLoss(model)
        self.result = torch.zeros(3).to(device)
        print("Hyp: ", self.loss_fn.hyp)
        
    def logging_loss(self, style="train"):
        if not(self.result is None):
            wandb.log({f"IoU_{style}" : self.result[0], f"BCE_{style}" : self.result[1], f"DFL_{style}" : self.result[2]})
        else:
            print("None")
            
    def clear_loss(self):
        self.result = torch.zeros(3).to(device)
        
    def forward(self, preds, targets):
        self.result += self.loss_fn(preds, targets)[0] 
        return self.result
    
class ComplexLoss(nn.Module):
    def __init__(self, model, device="cpu", weight = [1., 0.01], batch_size=5.0):
        super().__init__()
        self.weight = weight
        self.RGBTargetLoss = RGBTargetAwareLoss(device)
        self.YOLOLoss = YOLOv8Loss(model, batch_size).to(device)
        self.RGBResult, self.YOLOResults = None, None
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, output, target, preds, targets):
        self.RGBResult = self.RGBTargetLoss(output, target)
        self.YOLOResults = self.YOLOLoss(preds, targets)
        combined_loss = self.weight[0] * self.RGBResult + self.weight[1] * (self.YOLOResults).mean()
        return combined_loss.requires_grad_(True)
    
    def clear_loss(self):
        self.RGBTargetLoss.clear_loss()
        self.YOLOLoss.clear_loss()
    
    def logging_loss(self, style="train"):
        if not(self.RGBResult is None):
            self.RGBTargetLoss.logging_loss(style)
        if not(self.YOLOResults is None):
            self.YOLOLoss.logging_loss(style)

