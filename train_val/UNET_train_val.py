# Train'n'validate U-net model functions  

def train_unet(model, train_loader, optimizer, criterion, device, epoch):
    grad_norms = {}
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # print("I'm here! 1")
        images = images.to(device)
        print("Images size: ", images.size())
        optimizer.zero_grad()
        inter_images = normalize_pad(torch.squeeze(images, 1))
        patches_masked_images, num_patches_h, num_patches_w = create_patches(inter_images, (100, 100))
        _, outputs = model(patches_masked_images.requires_grad_(True))
        outputs, pad_images = patches_to_images(outputs, inter_images, num_patches_h, num_patches_w)
        outputs = (outputs).requires_grad_(True)
        print(outputs[0, :, 0, 0], pad_images[0, :, 0, 0]*255)
        loss = criterion(outputs, (pad_images[:, :3, :, :]).float())
        print("Loss: ", loss)
        loss.backward()
        gradients = []
        names = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        optimizer.step()
        running_loss += loss.item()
        if ((i + 1) % 10 == 0) and (i != len(train_loader)-1):
            print(f"Epoch [{epoch+1}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
            wandb.log({
                **{f"grad_norm/{name}": norm for name, norm in grad_norms.items()},
                "batch_loss": running_loss / 10
            })
            grad_norms = {}
            running_loss = 0.0
        if (i + 1) % 50 == 0:  # Реже, чтобы не перегружать
            gradients = []
            names = []
            for name, param in model.named_parameters():
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
    del images
    gc.collect()
    return running_loss 


def validate_unet(model, val_loader, criterion, device):
    model.eval()
    avg_loss = 0.0
    with torch.no_grad():
        for i, images in enumerate(val_loader):
            images = images.to(device)

            inter_images = normalize_pad(torch.squeeze(images, 1))
            masked_images = create_channel_masks(inter_images, inter_images.shape[0], inter_images.shape[2], inter_images.shape[3], device)
            patches_masked_images, num_patches_h, num_patches_w = create_patches(masked_images, (100, 100))

            _, outputs = model(patches_masked_images)

            outputs, pad_images = patches_to_images(outputs, inter_images, num_patches_h, num_patches_w)

            loss = criterion(outputs, (pad_images[:, :3, :, :]*255).float()) # Оценка потери
            print(loss)

            avg_loss += loss.item()
        del images
        gc.collect()

    return avg_loss / len(val_loader)

