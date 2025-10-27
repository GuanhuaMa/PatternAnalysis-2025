def train(model, train_loader, test_dataset, epochs=3, lr=0.001, visualize_every=1):


    criterion = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Train Starting for {epochs} epochs")

    for epoch in range(epochs):
        model.train() 

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            pred_pet = outputs[:, 0]  # (C, H, W)
            loss = criterion(pred_pet, masks)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pass 

        print(f"Epoch {epoch+1}/{epochs} Complete")

    print("Training complete with enhanced U-Net")
    return [] 