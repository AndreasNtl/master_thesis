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
        # Plot
        plt.style.use("dark_background")

        ax = self.df.label.value_counts().plot(kind='bar',
                                               stacked=True,
                                               figsize=(10, 6),
                                               color=["blue", "red"])

        ax.set_xticklabels(["Positive", "Negative"], rotation=45, fontsize=12)
        ax.set_ylabel('Total Images', fontsize=12)
        ax.set_title("Distribution of data grouped by label", fontsize=18, y=1.05)

        # Annotate
        for i, rows in enumerate(self.df.label.value_counts().values):
            ax.annotate(int(rows), xy=(i, rows - 12),
                        rotation=0, color="black", ha="center",
                        verticalalignment='bottom', fontsize=10, fontweight="bold")

        ax.text(1.2, 2550, f"Total {len(self.df)} images",
                size=15, color="black", ha="center", va="center",
                bbox=dict(boxstyle="round", fc="lightblue", ec="black", ))

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
