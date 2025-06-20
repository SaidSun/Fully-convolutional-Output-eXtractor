# # Constants

# ## Numbers

t_lower = 50
t_upper = 150
batch_size = 16
img_size = 1000
num_epochs = 5
best_val_loss = float('inf')
image_size = (640, 640)


# ## U-net params

levels = 4
channels = [[[5, 15, 30], [30, 60, 60], [60, 120, 120], [120, 120, 120]],
            [[[120//3, 80//3, 80//3], [120//3, 80//3, 80//3],[120//3, 80//3, 80//3],],
             [[(80//3+120//3//2), 32//3, 32//3],[(80//3+120//3//2), 32//3, 32//3],[(80//3+120//3//2), 32//3, 32//3],],
             [[(32//3+60//3//2), 32//3, 16//3],[(32//3+60//3//2), 32//3, 16//3],[(32//3+60//3//2), 32//3, 16//3],],
             [[(16//3+30//3//2), 8//3, 3//3], [(16//3+30//3//2), 8//3, 3//3],[(16//3+30//3//2), 8//3, 3//3]]]]
lvls_kernel = [[[5, 5],[5, 5], [3, 3], [3, 3]],[[3, 3], [3, 3], [5, 5], [5, 5]]]
pools = [2]*levels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ## Strings

depth_kind = "DA" # or "DA"

dtype = torch.float32

path_train_dataset = "/home/jupyter/datasphere/project/dataset-rgb-da/dataset_RGB_LABELS.pth"
path_val_dataset = "/home/jupyter/datasphere/project/dataset-rgb-da/dataset_RGB_val.pth"

model_depthDA = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

path_to_bench_data = "/home/jupyter/datasphere/project/dataset-benchmark/dataset_benchmark_RGB_LABELS.pth"

Checkpoint = "apple/DepthPro-hf"
image_processor = AutoImageProcessor.from_pretrained(Checkpoint, use_fast=True)
model_depthDP =  AutoModelForDepthEstimation.from_pretrained(
    Checkpoint, device_map=device, torch_dtype=dtype, use_fov_model=True
)


# ## Tensors


image_size = [640, 640]

