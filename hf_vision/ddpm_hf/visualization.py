import matplotlib.pyplot as plt

# Adapted from:
# https://huggingface.co/blog/annotated-diffusion
# which adapted it from https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py


def plot_images(
    images: list,
    original_images: list | None = None,
    row_title: str | None = None,
    **imshow_kwargs: dict,
) -> None:
    if not isinstance(images[0], list):
        images = [images]

    n_rows = len(images)
    n_cols = len(images[0]) + (1 if original_images is not None else 0)
    print(f"{n_rows=}, {n_cols=}")

    _, axs = plt.subplots(
        figsize=(n_cols * 5, n_rows * 5), nrows=n_rows, ncols=n_cols, squeeze=False
    )

    for row_idx, row in enumerate(images):
        with_original_image = (
            original_images is not None
            and len(original_images) > row_idx
            and original_images[row_idx] is not None
        )
        row = [original_images[row_idx]] + row if with_original_image else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(img, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if row_title is not None and col_idx == 0:
            ax.set_title(row_title)
        if with_original_image:
            axs[row_idx, 0].set_title("Original Image")
    plt.tight_layout()
