# U-NET class

class Block(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size=(1,1), g=1, block_type=None):
        super().__init__()
        self.block_type = block_type  # 'color', 'depth' или 'contour'
        
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size, groups=g),
            nn.BatchNorm2d(outChannels),
            self._get_activation()
        )
    def forward(self, x, use_checkpoint):
        if use_checkpoint:
            return checkpoint.checkpoint(self.conv, x)
        else:
            return self.conv(x)
    
    def _get_activation(self):
        if self.block_type == 'color':
            return nn.Sigmoid()
        elif self.block_type == 'depth':
            return nn.LeakyReLU(0.2)
        elif self.block_type == 'contour':
            return nn.Tanh()
        else:
            return nn.ReLU()

class ChannelAttention(nn.Module):
    def __init__(self, in_channels=30, num_groups=3, compress_ratio=2):
        super().__init__()

        self.num_groups = num_groups
        self.in_split = in_channels // num_groups
        self.out_split = in_channels // compress_ratio // num_groups

        # Слои для обработки каждой группы
        self.avg_pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(num_groups)])
        self.max_pools = nn.ModuleList([nn.AdaptiveMaxPool2d(1) for _ in range(num_groups)])

        # Блоки внимания для уменьшения каналов
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_split, self.in_split//compress_ratio, 1, bias=False),
                nn.BatchNorm2d(self.in_split//compress_ratio),
                nn.ReLU(),
                nn.Conv2d(self.in_split//compress_ratio, self.out_split, 1, bias=False),
                nn.BatchNorm2d(self.out_split),
                nn.Sigmoid()
            )
            for _ in range(num_groups)
        ])

        self.channel_reducers = nn.ModuleList([
            nn.Conv2d(self.in_split, self.out_split, 1, bias=False)
            for _ in range(num_groups)
        ])

    def forward(self, x):
        channels = torch.split(x, self.in_split, dim=1)
        outputs = []

        for i in range(self.num_groups):
            channel = channels[i]
            avg_att = self.attention[i](self.avg_pools[i](channel))
            max_att = self.attention[i](self.max_pools[i](channel))
            attention = torch.div((avg_att + max_att), 2).requires_grad_(True)

            reduced = self.channel_reducers[i](channel).requires_grad_(True)
            outputs.append(reduced * attention)

        return torch.cat(outputs, dim=1).requires_grad_(True)

class Encoder_lvl(nn.Module):
    def __init__(
        self,
        kernel_size,
        channels=(5, 15, 30, 42),
        g=1,
        pool=2,
        use_checkpoint=False,
        end=0,
    ):
        super().__init__()
        self.e = end
        self.poo = pool
        self.check = use_checkpoint
        
        self.encBlocks = nn.ModuleList(
            [
                Block(channels[i], channels[i + 1], kernel_size, g)
                for i in range(len(channels) - 2)
            ]
        )
        self.encBlocks = self.encBlocks.extend(
            [Block(channels[-2], channels[-1], (1, 1), g)]
        )
        
        self.pool = nn.MaxPool2d(self.poo, ceil_mode=True)
        self.attention_transf = ChannelAttention(
            in_channels=channels[-1], compress_ratio=2
        )

    def forward(self, x):
        # Создаем список, хранящий результат обработки каждым свёрточным блоком
        for block in self.encBlocks:
            x = block(x, self.check).requires_grad_(True)
        if not (self.e):
            if self.check:
                blockOutputs = checkpoint.checkpoint(self.attention_transf, x)
            else:
                blockOutputs = self.attention_transf(x)
        else:
            blockOutputs = []
        x = self.pool(x)
        return blockOutputs, x.requires_grad_(True)

class Decoder_lvl(nn.Module):
    def __init__(self, kernel_size, channels, g=1, pool=2, use_checkpoint=False, start_chan=128):
        super().__init__()
        self.check = use_checkpoint
        self.poo = pool

        # Три независимых блока для каждого канала
        self.color_block = self._make_block(channels[0], kernel_size, g, 'color')
        self.depth_block = self._make_block(channels[1], kernel_size, g, 'depth')
        self.contour_block = self._make_block(channels[2], kernel_size, g, 'contour')

        self.upconv = nn.ConvTranspose2d(
            start_chan, start_chan, kernel_size=8,
            stride=2, padding=0, output_padding=0)

    def _make_block(self, channels, kernel_size, groups, block_type):
        return nn.ModuleList([
            Block(channels[y], channels[y+1], kernel_size, 
                 g=groups, block_type=block_type)
            for y in range(len(channels)-1)
        ])

    def forward(self, x, enc_features):
        if isinstance(x, torch.Tensor):
            # Первый уровень - разделяем каналы на три части
            split_size = x.size(1) // 3
            color = x[:, :split_size, :, :]
            depth = x[:, split_size:2*split_size, :, :]
            contour = x[:, 2*split_size:, :, :]
        else:
            # Последующие уровни - получаем уже разделенные данные
            color, depth, contour = x
        
        # Апсемплинг каждой части отдельно
        color = checkpoint.checkpoint(self.upconv, color)
        depth = checkpoint.checkpoint(self.upconv, depth)
        contour = checkpoint.checkpoint(self.upconv, contour)

        # Добавление features от энкодера к каждой части
        if len(enc_features):
            enc_feat = self.crop(enc_features, color)
            enc_split = torch.split(enc_feat, enc_feat.size(1)//3, dim=1)
        
            # Обработка каждой ветки с уникальными фичами
            color = torch.cat([color, enc_split[0]], dim=1)
            depth = torch.cat([depth, enc_split[1]], dim=1)
            contour = torch.cat([contour, enc_split[2]], dim=1)
            del enc_feat
            del enc_split
            gc.collect()

        # Обработка каждого канала отдельно
        for block in self.color_block:
            color = block(color, self.check)

        for block in self.depth_block:
            depth = block(depth, self.check)

        for block in self.contour_block:
            contour = block(contour, self.check)

        return (color, depth, contour)

    def crop(self, enc_features, x):
        _, _, H, W = x.shape
        return tr.CenterCrop((H, W))(enc_features)

class Encoder(nn.Module):
    def __init__(self, levels: int, channels, pool, kernel_size, g=None, use_checkpoint=False): #channels, pool и kernel_size - это всё списки, которые хранят информацию для каждого уровня энкодера отдельно
        super().__init__()
        if g == None:
            self.groups = [1]*levels
        else:
            self.groups = g
        self.lvls = nn.ModuleList(
        [Encoder_lvl(kernel_size=kernel_size[i], channels=channels[i], g=self.groups[i], pool=pool[i], use_checkpoint=use_checkpoint)
         if i < levels-1 else Encoder_lvl(kernel_size=kernel_size[i], channels=channels[i], g=self.groups[i], pool=pool[i], use_checkpoint=use_checkpoint, end = 1)
         for i in range(levels)])

    def forward(self, x):
        blockOutputs = []
        for i, lvl in enumerate(self.lvls):
            blk, x = lvl(x)
            blockOutputs.append(blk)
        return blockOutputs, x.requires_grad_(True)

class Decoder(nn.Module):
    def __init__(self, levels: int, channels, pool, kernel_size, g=None, use_checkpoint=False):
        super().__init__()
        if g is None:
            self.groups = [1] * levels
        else:
            self.groups = g
        self.lvls = nn.ModuleList(
            [Decoder_lvl(kernel_size=kernel_size[i], channels=channels[i], g=self.groups[i],
                         pool=pool[i], use_checkpoint=use_checkpoint,
                         start_chan=channels[i][0][0]) if i == 0 else
             Decoder_lvl(kernel_size=kernel_size[i], channels=channels[i], g=self.groups[i],
                         pool=pool[i], use_checkpoint=use_checkpoint,
                         start_chan=channels[i-1][-1][-1])
             for i in range(levels)])

    def forward(self, x, blockOutputs):
        for i, blck in enumerate(reversed(blockOutputs)):
            x = self.lvls[i](x, blck)

        color, depth, contour = x
        return torch.cat([color, depth, contour], dim=1).requires_grad_(True)

class UNET(nn.Module):
    def __init__(self, levels, channels, lvls_kernel, pools=None, gs=None, use_checkpoint=True):
        super().__init__()
        print(use_checkpoint)
        if gs==None:
            self.groups = [1]*levels
        else:
            self.groups = [1]*levels
        if pools==None:
            self.poos = [2]*levels
        else:
            self.poos = pools
        self.encoder = Encoder(levels, channels[0], self.poos, lvls_kernel[0], self.groups, use_checkpoint)
        self.decoder = Decoder(levels, channels[1], self.poos[::-1], lvls_kernel[1], self.groups[::-1], use_checkpoint)
        self.initialize_weights()
        
    def logging_grads_info(self, logging_wandb):
        gradients = []
        names = []
        for name, param in self.encoder.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.flatten())
                names.append(name)
        for name, param in self.decoder.named_parameters():
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

    def get_color_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'color_block' in name:
                params.append(param)
        return params
    
    def get_depth_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'depth_block' in name:
                params.append(param)
        return params
    
    def get_contour_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'contour_block' in name:
                params.append(param)
        return params
    
    def get_shared_params(self):
        params = []
        for name, param in self.named_parameters():
            if not any(x in name for x in ['color_block', 'depth_block', 'contour_block']):
                params.append(param)
        return params
    
    def initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # Разные стратегии инициализации для разных веток
                if 'color_block' in name:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif 'depth_block' in name:
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('sigmoid'))
                elif 'contour_block' in name:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # Небольшой bias для разнообразия

    def forward(self, x):
        block_outputs, x = self.encoder(x)
        x = self.decoder(x, block_outputs)
        # x = torch.clamp(x, 0.001, 1.0)
        return (x).requires_grad_(True)

