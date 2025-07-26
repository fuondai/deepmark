import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import quantization as q_models
from torch.utils.data import DataLoader
import copy
import os

# Import our custom modules
from deepmark import DeepMark
from attacks import finetune_model, prune_model, quantize_model
# --- Configuration ---
MODEL_ARCH = "resnet18"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SECRET_KEY = "" # TODO: Set your secret key here
WRONG_SECRET_KEY = "" # TODO: Set your wrong secret key here
BATCH_SIZE = "" # TODO: Set your Batch Size here
LEARNING_RATE = "" # TODO: Set your Learning Rate here
TRAIN_EPOCHS = "" # TODO: Set your Train Epochs here
FINETUNE_EPOCHS = "" # TODO: Set your Fine-tune Epochs here

# Watermark parameters from the paper
WM_BITS = "" # TODO: Set your WM Bits here
WM_LAMBDA = "" # TODO: Set your WM Lambda here
TARGET_LAYER_NAME = 'fc'

# --- Data Loading ---
def get_data_loaders():
    print("Loading CIFAR-10 data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Loader for calibration in quantization attack
    calibration_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, calibration_loader

# --- Model Training and Evaluation ---
def evaluate(model, data_loader, model_name="Model", force_cpu=False):
    # If force_cpu is True (for quantized models), run on CPU. Otherwise, use the default device.
    eval_device = torch.device("cpu") if force_cpu else DEVICE

    model.to(eval_device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(eval_device), labels.to(eval_device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy of {model_name}: {accuracy:.2f} %")
    return accuracy

def train(model, train_loader, test_loader, epochs, lr, deepmark_manager=None, wm_lambda=0.0):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    watermark_loss_fn = None
    if deepmark_manager:
        watermark_loss_fn = deepmark_manager.get_watermark_loss_fn(DEVICE)
        print(f"Starting training with watermark (lambda={wm_lambda})...")
    else:
        print("Starting baseline training (no watermark)...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            task_loss = criterion(outputs, labels)
            total_loss = task_loss

            if watermark_loss_fn:
                target_layer_weights = dict(model.named_modules())[TARGET_LAYER_NAME].weight
                wm_loss = watermark_loss_fn(target_layer_weights)
                total_loss = task_loss + (wm_lambda * wm_loss)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        evaluate(model, test_loader, "Intermediate")

    print("Training finished.")
    return model

def run_ablation_study(train_loader, test_loader, calibration_loader, lambda_values, bits_values):
    ablation_results = []

    for wm_lambda in lambda_values:
        for wm_bits in bits_values:
            print(f"\n{'='*50}")
            print(f"RUNNING ABLATION: lambda={wm_lambda}, bits={wm_bits}")
            print(f"{'='*50}\n")

            # Train Baseline Model (only once per model architecture, but for simplicity, re-train for each ablation point)
            # In a real scenario, you'd train the baseline once and reuse it.
            baseline_model = q_models.resnet18(weights=None, progress=True, quantize=False)
            baseline_model.fc = nn.Linear(baseline_model.fc.in_features, 10)
            baseline_model = train(baseline_model, train_loader, test_loader, TRAIN_EPOCHS, LEARNING_RATE)
            acc_base = evaluate(baseline_model, test_loader, "Baseline Model")

            # Train Watermarked Model
            watermarked_model_pristine = q_models.resnet18(weights=None, progress=True, quantize=False)
            watermarked_model_pristine.fc = nn.Linear(watermarked_model_pristine.fc.in_features, 10)
            target_layer = dict(watermarked_model_pristine.named_modules())[TARGET_LAYER_NAME]
            deepmark_manager = DeepMark(target_layer, secret_key=SECRET_KEY, watermark_bits=wm_bits)
            
            watermarked_model_pristine = train(
                watermarked_model_pristine, train_loader, test_loader, 
                TRAIN_EPOCHS, LEARNING_RATE, deepmark_manager, wm_lambda
            )
            acc_wm = evaluate(watermarked_model_pristine, test_loader, "Watermarked Model")

            # Store fidelity results
            fidelity_result = {
                "lambda": wm_lambda,
                "bits": wm_bits,
                "acc_baseline": acc_base,
                "acc_watermarked": acc_wm,
                "acc_drop": acc_base - acc_wm
            }

            # =========================================================================
            # 2. ROBUSTNESS EVALUATION
            # =========================================================================
            print("\n" + "="*50)
            print("RUNNING EXPERIMENT 2: ROBUSTNESS EVALUATION")
            print("="*50)

            # --- Verification Helper ---
            def verify_and_check(model, manager, model_name):
                p_value = manager.verify(model, TARGET_LAYER_NAME)
                detected = p_value <= 0.05 
                print(f"Verification for {model_name}: p-value={p_value:.4f}, Detected={detected}")
                return p_value, detected

            # No Attack
            p_val_none, detected_none = verify_and_check(watermarked_model_pristine, deepmark_manager, "Pristine Watermarked")
            
            # Pruning Attack
            pruned_model = prune_model(watermarked_model_pristine, amount=0.2)
            acc_prune = evaluate(pruned_model, test_loader, "Pruned Model")
            p_val_prune, detected_prune = verify_and_check(pruned_model, deepmark_manager, "Pruned Model")

            # Fine-tuning Attack
            finetuned_model = finetune_model(watermarked_model_pristine, train_loader, epochs=FINETUNE_EPOCHS, device=DEVICE)
            acc_finetune = evaluate(finetuned_model, test_loader, "Fine-tuned Model")
            p_val_finetune, detected_finetune = verify_and_check(finetuned_model, deepmark_manager, "Fine-tuned Model")

            # Quantization Attack
            quantized_model = quantize_model(watermarked_model_pristine.to('cpu'), calibration_loader)
            acc_quant = evaluate(quantized_model, test_loader, "Quantized Model", force_cpu=True)
            p_val_quant, detected_quant = verify_and_check(quantized_model, deepmark_manager, "Quantized Model")

            robustness_results = {
                "no_attack": {"acc": acc_wm, "p_value": p_val_none, "detected": str(detected_none)},
                "pruning": {"acc": acc_prune, "p_value": p_val_prune, "detected": str(detected_prune)},
                "fine_tuning": {"acc": acc_finetune, "p_value": p_val_finetune, "detected": str(detected_finetune)},
                "quantization": {"acc": acc_quant, "p_value": p_val_quant, "detected": str(detected_quant)}
            }

            # =========================================================================
            # 3. UNFORGEABILITY EVALUATION
            # =========================================================================
            print("\n" + "="*50)
            print("RUNNING EXPERIMENT 3: UNFORGEABILITY EVALUATION")
            print("="*50)

            # Verify baseline model with correct key (should fail)
            p_val_base, detected_base = verify_and_check(baseline_model, deepmark_manager, "Baseline with Correct Key")

            # Verify watermarked model with correct key (should succeed)
            p_val_wm_correct, detected_wm_correct = verify_and_check(watermarked_model_pristine, deepmark_manager, "Watermarked with Correct Key")

            # Verify watermarked model with wrong key (should fail)
            imposter_manager = DeepMark(target_layer, secret_key=WRONG_SECRET_KEY, watermark_bits=wm_bits)
            p_val_wm_wrong, detected_wm_wrong = verify_and_check(watermarked_model_pristine, imposter_manager, "Watermarked with Wrong Key")

            unforgeability_results = {
                "baseline_correct_key": {"p_value": p_val_base, "detected": str(detected_base)},
                "watermarked_correct_key": {"p_value": p_val_wm_correct, "detected": str(detected_wm_correct)},
                "watermarked_wrong_key": {"p_value": p_val_wm_wrong, "detected": str(detected_wm_wrong)}
            }
            
            ablation_results.append({
                "fidelity": fidelity_result,
                "robustness": robustness_results,
                "unforgeability": unforgeability_results
            })
    return ablation_results

def main():
    train_loader, test_loader, calibration_loader = get_data_loaders()

    lambda_values = [0.01, 0.1, 0.5]
    bits_values = [256, 512, 1024, 2048]

    results = run_ablation_study(train_loader, test_loader, calibration_loader, lambda_values, bits_values)

    import json
    print("\n" + "="*50)
    print("ABLATION STUDY COMPLETE - RESULTS:")
    print("="*50)
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()
