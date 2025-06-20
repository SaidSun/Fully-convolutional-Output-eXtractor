class TensorDataset(Dataset):
    def __init__(self, data_tensor, depth_kind="DA"):
        self.depth_kind = depth_kind
        self.data_tensor = data_tensor
        self.data_length = len(data_tensor["cls"]) 
        
        if self.data_length == 0:
            raise ValueError("Data tensor cannot be empty.")
    
    
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return {"image": five_channels(self.data_tensor["images"][idx], self.depth_kind), "cls": self.data_tensor["cls"][idx], "boxes" : self.data_tensor["bboxes"][idx]}


# Dataset preprocess functions


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    clss = [item['cls'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,  # Список тензоров разной длины
        'clss': clss  # Список меток
    }

