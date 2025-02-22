import cv2
import numpy as np
from mpl_toolkits.axisartist.axes_grid import ImageGrid
import matplotlib.pyplot as plt


class Visualization:

    def __init__(self, df):
        self.df = df

    def plot_images_and_masks(self, img_size=128):
        # Data
        sample_df = self.df[self.df["label"] == 1].sample(5).values
        sample_imgs = []
        for i, data in enumerate(sample_df):
            img = cv2.resize(cv2.imread(data[0]), (img_size, img_size))
            mask = cv2.resize(cv2.imread(data[1]), (img_size, img_size))
            sample_imgs.extend([img, mask])

        sample_imgs_arr = np.hstack(np.array(sample_imgs[::2]))
        sample_masks_arr = np.hstack(np.array(sample_imgs[1::2]))

        # Plot
        fig = plt.figure(figsize=(25., 25.))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.1,)

        grid[0].imshow(sample_imgs_arr)
        grid[0].set_title("Images", fontsize=15)
        grid[0].axis("off")
        grid[1].imshow(sample_masks_arr)
        grid[1].set_title("Masks", fontsize=15, y=0.9)
        grid[1].axis("off")
        plt.show()

    def plot_distribution_grouped_by_label(self):
        plt.style.use("dark_background")

        ax = self.df.label.value_counts().plot(kind='bar',
                                               figsize=(10, 6),
                                               color=["red", "blue"],
                                               edgecolor='white',
                                               alpha=0.8)

        ax.set_xticklabels(["Positive", "Negative"], rotation=0, fontsize=12)
        ax.set_ylabel('Total Images', fontsize=12)
        ax.set_title("Distribution of Data by Label", fontsize=18, y=1.05)

        total_images = len(self.df)
        for i, rows in enumerate(self.df.label.value_counts().values):
            percentage = (rows / total_images) * 100
            ax.annotate(f"{int(percentage)}%", xy=(i, rows + 10),
                        rotation=0, color="white", ha="center",
                        fontsize=10, fontweight="bold")

        ax.text(0.5, 0.95, f"Total: {total_images} Images",
                size=15, color="white", ha="center", va="center",
                bbox=dict(boxstyle="round", fc="red", ec="black", alpha=0.7))

        # Add legend with labels specified
        ax.legend(labels=["Positive", "Negative"], loc="upper right")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()


    def plot_images(self, inputs, nrows=5, ncols=5):
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(wspace=0., hspace=0.)
        i_ = 0

        if len(inputs) > 25:
            inputs = inputs[:25]

        for idx in range(len(inputs)):
            img = (inputs[idx] * 127.5) + 127.5
            # Convert image back to uint8
            img = img.numpy().astype(np.uint8)

            plt.subplot(nrows, ncols, i_ + 1)
            plt.imshow(img)
            plt.axis('off')

            i_ += 1

        return plt.show()
# %%
