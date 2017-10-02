
import matplotlib.pyplot as plt

def plot_images(images):
    fig, ax = plt.subplots()
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap="gray")
    plt.show()
