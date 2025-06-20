# Train'n'val process

def train_val_model(model, num_epochs, train_loader, val_loader, logging_wandb, conf, proj, loss_func, train_func, val_func ):
    logging_wandb.start(proj, conf)
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)
    loss_func.clear_loss()
    last_loss = 0.0
    for epoch in range(num_epochs):
        for i, batch_train in enumerate(train_loader):
            model.train()
            print("Result: ", loss_func.result)
            # if i < len(train_loader):
            train_loss = train_func(model, i, epoch, batch_train, loss_func, len_train_loader, logging_wandb)
                # last_loss += train_loss
            # else:
            #     train_func(model, i, epoch, batch_train, loss_func, len_train_loader, logging_wandb, is_last=True, last_loss=last_loss)
            #     last_loss = 0.0
            loss_func.clear_loss()
            del train_loss
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.test()
                for j, batch_val in enumerate(val_loader):
                    if j < len(val_loader)-1:
                        val_func(model, i, epoch, batch_val, loss_func, len_val_loader, logging_wandb, is_last=False)
                    else:
                        print("LAST ONE!")
                        val_func(model, i, epoch, batch_val, loss_func, len_val_loader, logging_wandb, is_last=True)
                    torch.cuda.empty_cache()
                loss_func.clear_loss()
    logging_wandb.fin()

