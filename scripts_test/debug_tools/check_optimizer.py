#!/usr/bin/env python3
"""
Check optimizer state from checkpoint.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

def main():
    checkpoint_path = Path(__file__).parent.parent / "models" / "best_model.pt"

    if not checkpoint_path.exists():
        print("No checkpoint found")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("Checkpoint keys:", list(checkpoint.keys()))

    if 'optimizer_state_dict' in checkpoint:
        optimizer_state = checkpoint['optimizer_state_dict']
        print("\nOptimizer state keys:", list(optimizer_state.keys()))

        if 'param_groups' in optimizer_state:
            print(f"\nNumber of parameter groups: {len(optimizer_state['param_groups'])}")
            for i, group in enumerate(optimizer_state['param_groups']):
                print(f"\nGroup {i}:")
                for key, value in group.items():
                    if key != 'params':
                        print(f"  {key}: {value}")

        if 'state' in optimizer_state:
            print(f"\nOptimizer state contains {len(optimizer_state['state'])} parameter states")

            # Check first few parameters
            count = 0
            for param_id, state in optimizer_state['state'].items():
                print(f"\nParameter {param_id}:")
                for key, value in state.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: shape={value.shape}, mean={value.mean().item():.6f}, std={value.std().item():.6f}")
                    else:
                        print(f"  {key}: {value}")
                count += 1
                if count >= 3:
                    print(f"... (showing first 3 of {len(optimizer_state['state'])} parameters)")
                    break

    # Check model parameters gradients
    if 'model_state_dict' in checkpoint:
        print("\n\nChecking model parameter statistics:")
        model_state = checkpoint['model_state_dict']

        # Group by layer type
        layers = {}
        for name, param in model_state.items():
            layer_type = name.split('.')[0] if '.' in name else 'other'
            if layer_type not in layers:
                layers[layer_type] = []
            layers[layer_type].append((name, param))

        for layer_type, params in layers.items():
            print(f"\n{layer_type}:")
            for name, param in params[:3]:  # Show first 3
                print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            if len(params) > 3:
                print(f"  ... and {len(params)-3} more parameters")

    # Check step
    if 'step' in checkpoint:
        print(f"\nStep: {checkpoint['step']}")

if __name__ == "__main__":
    main()