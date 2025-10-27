def train(model, train_loader, test_dataset, epochs=3, lr=0.001, visualize_every=1):

    print(f"Train Starting for {epochs} epochs")

    for epoch in range(epochs):
        model.train() 

        # TODO: batch loop

        # 打印一下这轮的训练成果
        print(f"Epoch {epoch+1}/{epochs} Complete")

    print("Train")
    return [] # 暂时返回一个空列表