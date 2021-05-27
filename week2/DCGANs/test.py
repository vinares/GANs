import numpy as np
import matplotlib.pyplot as plt
G_loss = [0.4,0.2,0.3]
D_loss = [0.7,0.6,0.9]
print(G_loss)
print(D_loss)
plt.figure()
plt.plot(np.arange(len(G_loss)) * 1000, G_loss, label='Generator Loss', lw=2)
plt.plot(np.arange(len(D_loss)) * 1000, D_loss, label='Discriminator Loss', lw=2)
plt.legend()
plt.xlim(0, len(D_loss) * 1000 + 2000)
plt.ylim(0, 1)
plt.xlabel('Current Steps')
plt.ylabel('Loss')
plt.title('Loss of Generator and Discriminator')
plt.show()