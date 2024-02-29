import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(f"{x_train.shape=} {y_train.shape=}")
print(f"{x_test.shape=} {y_test.shape=}")

labels = {0: "T-shirt/top",
          1: "Trouser",
          2: "Pullover",
          3: "Dress",
          4: "Coat",
          5: "Sandal",
          6: "Shirt",
          7: "Sneaker",
          8: "Bag",
          9: "Ankle boot"}

wandb.init(project="CS6910-Assignment1")

# s_img, s_lab = [], []
n_labels = len(labels)
fig = plt.figure(figsize=(n_labels,1))
for i in range(n_labels):
    idx = np.random.choice(np.where(y_train==i)[0], 1).item()
    plt.subplot(1, n_labels, i+1)
    plt.imshow(x_train[idx].reshape(28,28), cmap="gray")
    plt.xlabel(labels[y_train[idx]])
    plt.xticks([])
    plt.yticks([])
    # s_img.append(x_train[idx].reshape(28,28))
    # s_lab.append(labels[y_train[idx]])

plt.savefig("sample.png")

# wandb.log({"sample": [wandb.Image(img, caption=cap) for img, cap in zip(s_img, s_lab)]})
wandb.log({"sample": fig})
wandb.finish()

