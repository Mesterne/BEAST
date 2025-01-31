import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_model(model, X, y, batch_size, num_epochs, learning_rate):
    # Convert data to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    model.loss = torch.nn.L1Loss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []

    for _ in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, targets)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)

    model.save_model()

    return loss_history


def run_model_inference(model, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model to evaluation mode
    model.eval()

    # Convert test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Get predictions
        outputs = model(X_test_tensor)
        predicted = outputs.cpu().numpy()

    return predicted