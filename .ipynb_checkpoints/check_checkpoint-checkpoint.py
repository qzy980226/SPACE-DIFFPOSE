import torch

checkpoint_path = 'exp/speedplus_v2_diffpose_uvxyz_gt/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"Checkpoint structure:")
print(f"  Type: {type(checkpoint)}")
print(f"  Length: {len(checkpoint)}")
print()

model_dict = checkpoint[0]
print(f"Model state_dict has {len(model_dict)} keys")
print()

# Check for visibility-related keys
vis_keys = [k for k in model_dict.keys() if 'visibility' in k.lower()]
print(f"Visibility-related keys: {len(vis_keys)}")
if vis_keys:
    for k in vis_keys:
        print(f"  - {k}: {model_dict[k].shape}")
else:
    print("  None found!")
print()

# Show first 10 keys
print("First 10 keys in model_dict:")
for i, key in enumerate(list(model_dict.keys())[:10]):
    print(f"  {i+1}. {key}")
