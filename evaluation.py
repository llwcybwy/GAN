import CMMD
import os

train_path = "Data/val/Monet"
evaluation_path = "Output/Monet/10"

best_path = None
max_cmmd_value = float('-inf')  # Initialize to the lowest possible value

for item in os.listdir(evaluation_path):
    full_path = os.path.join(evaluation_path, item)
    print(full_path)  # Prints file and folder names
    print("Computing CMMD")

    cmmd_value = CMMD.compute_cmmd(train_path, full_path)
    print(f"CMMD Value: {cmmd_value}")

    if cmmd_value > max_cmmd_value:
        max_cmmd_value = cmmd_value
        best_path = full_path

print(f"Path with highest CMMD value: {best_path} ({max_cmmd_value})")