import torch
import torch.nn as nn
from scipy.stats import ttest_ind

class WatermarkLoss(nn.Module):
    """Calculates the watermark loss based on the DeepMark paper.

    The loss is the Mean Squared Error between the projected weights of a target
    layer and a secret watermark pattern.
    """
    def __init__(self, projection_matrix, watermark_pattern):
        super(WatermarkLoss, self).__init__()
        if not isinstance(projection_matrix, torch.Tensor):
            raise TypeError("projection_matrix must be a torch.Tensor")
        if not isinstance(watermark_pattern, torch.Tensor):
            raise TypeError("watermark_pattern must be a torch.Tensor")
        
        self.projection_matrix = projection_matrix
        self.watermark_pattern = watermark_pattern
        self.mse_loss = nn.MSELoss()

    def forward(self, layer_weights):
        """Calculates the forward pass for the watermark loss."""
        # Flatten the weights and project them
        projected_weights = torch.matmul(self.projection_matrix, layer_weights.view(-1))
        # Calculate MSE between projected weights and the secret pattern
        loss = self.mse_loss(projected_weights, self.watermark_pattern)
        return loss

class DeepMark:
    """Manages the DeepMark watermarking process: generation, embedding, and verification."""
    def __init__(self, target_layer, secret_key=1234, watermark_bits=256):
        """
        Args:
            target_layer (torch.nn.Module): The neural network layer to embed the watermark in.
            secret_key (int): The secret key to seed the random number generator.
            watermark_bits (int): The length of the watermark bitstring (L in the paper).
        """
        if not isinstance(target_layer, nn.Module):
            raise TypeError("target_layer must be a torch.nn.Module")
        
        self.target_layer = target_layer
        self.secret_key = secret_key
        self.watermark_bits = watermark_bits
        
        # Get the number of parameters (weights) in the target layer
        if isinstance(self.target_layer, nn.Sequential):
            # Assuming the linear layer is the last one in the Sequential module
            self.num_params = self.target_layer[-1].weight.numel()
        else:
            self.num_params = self.target_layer.weight.numel()
        
        # Generate the projection matrix and watermark pattern
        self.projection_matrix, self.watermark_pattern = self._generate_components()

    def _generate_components(self):
        """Generates the projection matrix and watermark pattern using the secret key."""
        # Use the secret key to ensure reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.secret_key)

        # 1. Generate Projection Matrix (P in the paper)
        # Shape: (watermark_bits, num_params)
        proj_matrix = torch.randn(
            (self.watermark_bits, self.num_params), 
            generator=generator
        )
        # Normalize each row to have a unit L2-norm as specified
        proj_matrix = torch.nn.functional.normalize(proj_matrix, p=2, dim=1)

        # 2. Generate Watermark Pattern (W in the paper)
        # Shape: (watermark_bits,)
        watermark_pattern = (torch.randint(0, 2, (self.watermark_bits,), generator=generator) * 2 - 1).float()
        
        return proj_matrix, watermark_pattern

    def get_watermark_loss_fn(self, device):
        """Returns the watermark loss function ready for training."""
        # Move components to the correct device (e.g., 'cuda')
        proj_matrix = self.projection_matrix.to(device)
        wm_pattern = self.watermark_pattern.to(device)
        return WatermarkLoss(proj_matrix, wm_pattern)

    def verify(self, suspect_model, target_layer_name):
        """
        Verifies the presence of the watermark in a suspect model.

        Args:
            suspect_model (torch.nn.Module): The model to be verified.
            target_layer_name (str): The name of the layer to extract weights from.

        Returns:
            float: The p-value from the paired t-test.
        """
        # Now, proceed with verification on the clean, non-quantized model
        try:
            suspect_layer = dict(suspect_model.named_modules())[target_layer_name]
            
            # If the layer is a quantized linear layer, get its de-quantized weight tensor
            if isinstance(suspect_layer, torch.ao.nn.quantized.Linear):
                suspect_weights = suspect_layer.weight().dequantize().clone().detach().cpu()
            # If it's a regular float layer, or another type of quantized layer where .weight is a tensor
            else:
                suspect_weights = suspect_layer.weight.clone().detach().cpu()
        except KeyError:
            raise ValueError(f"Layer '{target_layer_name}' not found in the suspect model.")

        # Use torch.mv for explicit matrix-vector multiplication
        projected_weights = torch.mv(
            self.projection_matrix.cpu(), 
            suspect_weights.view(-1)
        )

        # Perform the independent two-sample t-test
        # Partition projected_weights into two groups based on watermark_pattern
        group_plus_1 = projected_weights[self.watermark_pattern.cpu() == 1]
        group_minus_1 = projected_weights[self.watermark_pattern.cpu() == -1]

        # Perform independent t-test, expecting mean of group_plus_1 to be greater
        t_statistic, p_value = ttest_ind(group_plus_1.numpy(), group_minus_1.numpy(), equal_var=False, alternative='greater')
        
        return p_value

# Example Usage (for demonstration)
if __name__ == '__main__':
    # 1. Create a dummy model and select a target layer
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10) # Target layer
    )
    target_layer = model[2]

    # 2. Initialize the DeepMark manager
    deepmark = DeepMark(target_layer, secret_key=1234, watermark_bits=64)

    # 3. Get the loss function for training
    # In a real scenario, you would add this to your main training loss
    watermark_loss_fn = deepmark.get_watermark_loss_fn(device='cpu')
    current_weights = target_layer.weight
    loss = watermark_loss_fn(current_weights)
    print(f"Initial Watermark Loss: {loss.item()}")

    # 4. Simulate verification on the original model
    # We expect a very high p-value since the weights haven't been trained with the loss
    # (This is just a sanity check of the mechanism)
    p_value_before = deepmark.verify(model, '2')
    print(f"P-value (before training): {p_value_before:.4f}")
    print("A low p-value is expected here, as the watermark is not yet embedded.")

    # 5. Simulate verification with a different key (should fail)
    imposter_deepmark = DeepMark(target_layer, secret_key="", watermark_bits=64) # TODO: Set your secret key here
    p_value_wrong_key = imposter_deepmark.verify(model, '2')
    print(f"P-value (wrong key): {p_value_wrong_key:.4f}")
    print("This should also be low, showing unforgeability.")

    # After training a model with the watermark loss, the p-value for the correct
    # key should become very high (e.g., > 0.05), confirming ownership.
