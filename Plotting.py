import matplotlib.pyplot as plt
import numpy as np

epochs = [1, 2, 3, 4, 5, 6,7 ,8 ,9, 10]

# Given data
G_loss = [0.47846911786632107, 0.49271671184423294, 0.48296554513614287, 0.43586575146764517, 0.4211772814054381, 0.4335728627883575, 0.4120156628164378, 0.3988547422499819,0.3838171498511325, 0.37331964055245576]
D_loss = [4.321077000011098, 3.9388912757689303, 3.4744523108005523, 3.4144549159841104, 3.3707821189002556, 3.196422424709255, 3.13757202422077, 3.100845082375136, 2.9472982627424327, 2.994682861864567]
CMMD_Monet = [3.2254457, 3.1461716, 3.058672, 3.0003786, 2.9523373, 2.8909445, 2.8705597, 2.873063, 2.8426647, 2.8367043]
CMMD_Photo = [3.541112, 3.3906698, 3.3165216, 3.2479763, 3.1362772, 3.0645132, 2.9428005, 2.8412342, 2.8584003, 2.8625727]

# Create first plot for G_loss and D_loss with secondary y-axis for G_loss
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(epochs, D_loss, marker='s', linestyle='--', linewidth=1.5, color='blue', label='Discriminator')
ax2.plot(epochs, G_loss, marker='o', linestyle='--', linewidth=1.5, markersize=8, color='red', label='Generator')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Discriminator loss', color='blue')
ax2.set_ylabel('Generator loss', color='red')
ax1.set_title('Generator and discriminator Loss')
ax1.set_xticks(epochs)
# Adjust legend placement to prevent overlap
ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.90))

ax1.grid()
plt.show()

# Create second plot for CMMD_Monet and CMMD_Photo
plt.figure(figsize=(10, 5))
plt.plot(epochs, CMMD_Monet, marker='^', linestyle='--', linewidth=1.5, markersize=7, label=' Monet')
plt.plot(epochs, CMMD_Photo, marker='v', linestyle='--', linewidth=1.5, markersize=7, label=' photo')
plt.xlabel('Epochs')
plt.ylabel('CMMD score')
plt.title('CMMD for generated Monet\'s and photos')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.xticks(epochs)
plt.grid()
plt.show()
