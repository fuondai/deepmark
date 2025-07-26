import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import copy
import torchvision

def finetune_model(model, train_loader, epochs=5, learning_rate=1e-4, device='cuda'):
    """
    Fine-tunes a given model for a few epochs.

    Args:
        model (nn.Module): The model to be fine-tuned.
        train_loader (DataLoader): DataLoader for the training data.
        epochs (int): Number of epochs for fine-tuning.
        learning_rate (float): Learning rate for the optimizer.
        device (str): The device to train on ('cuda' or 'cpu').

    Returns:
        nn.Module: The fine-tuned model.
    """
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    # Create a deep copy to avoid modifying the original model in place
    finetuned_model = copy.deepcopy(model).to(device)
    finetuned_model.train()
    
    optimizer = optim.Adam(finetuned_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = finetuned_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Fine-tuning Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    print("Fine-tuning complete.")
    return finetuned_model

def prune_model(model, amount=0.2):
    """
    Applies magnitude-based pruning to the model's layers.

    Args:
        model (nn.Module): The model to be pruned.
        amount (float): The fraction of connections to prune (e.g., 0.2 for 20%).

    Returns:
        nn.Module: The pruned model.
    """
    print(f"\nApplying pruning with {amount*100:.0f}% sparsity...")
    pruned_model = copy.deepcopy(model)
    
    # Prune all convolutional and linear layers
    for module in pruned_model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make the pruning permanent by removing the re-parameterization
            prune.remove(module, 'weight')
            
    print("Pruning complete.")
    return pruned_model

def quantize_model(model, calibration_loader):
    """
    Applies post-training static quantization to a quantization-aware model.

    Args:
        model (nn.Module): A quantization-ready model (e.g., from torchvision.models.quantization).
        calibration_loader (DataLoader): DataLoader with data for calibration.

    Returns:
        nn.Module: The quantized INT8 model.
    """
    print("\nApplying post-training static quantization...")
    quantized_model = copy.deepcopy(model).to('cpu')
    quantized_model.eval()

    # The model from torchvision.models.quantization already has a qconfig.
    # We just need to specify the backend.
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Fuse modules - the quantization-aware model has a fuse_model() method
    quantized_model.fuse_model()
    
    # Prepare and calibrate
    torch.quantization.prepare(quantized_model, inplace=True)
    print("Calibrating...")
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            quantized_model(inputs)
    
    # Convert to quantized version
    torch.quantization.convert(quantized_model, inplace=True)
    print("Quantization complete.")
    return quantized_model

