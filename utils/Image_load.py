# Image load functions

def load_images(k, key=None):
    data_torch = None
    flag = False
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            filepath = os.path.join(dirname, filename) 
            if key is None:
                is_RGBDE = not("RGBDE" in filepath)
                type_image = "image_RGBD"
            else:
                is_RGBDE = ("RGBDE" in filepath)
                type_image = "image_RGBDE"
            if (".pkl" in filepath) and is_RGBDE and not("length" in filepath):
                print(filepath)
                with open(filepath,  'rb') as file:
                    if k < 950:
                        new_torch = nn.functional.interpolate(torch.stack(pickle.load(file)[type_image][k-steps:k]), size=(1024, 2048), mode='bilinear', align_corners=False)
                    else:
                        try:
                            new_torch = nn.functional.interpolate(torch.stack(pickle.load(file)[type_image][k-steps:k]), size=(1024, 2048), mode='bilinear', align_corners=False)
                        except:
                            new_torch = None
                    if new_torch is not None:
                        if data_torch is None:
                            data_torch = new_torch
                        else:
                            data_torch = torch.cat((data_torch, new_torch), dim=0)
                    del new_torch
                    gc.collect()
                    print("Clean")
    return data_torch

