import os
import io
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class MediaSaver:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _to_uint8(self, img):
        img = np.asarray(img)
        if img.dtype == np.uint8:
            return img
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    def _render_plot(
        self,
        values_so_far,
        ground_truth_so_far,
        total_len,
        height,
        width=500,
        ymin=None,
        ymax=None,
        title="Value over time",
        stride=1,
    ):
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        x = np.arange(len(values_so_far)) * int(stride)
        ax.plot(x, values_so_far, linewidth=2, label="predicted")
        if ground_truth_so_far is not None and len(ground_truth_so_far) > 0:
            ax.plot(x, ground_truth_so_far, linewidth=2, linestyle="--", label="ground_truth")
        ax.set_xlim(0, max(total_len - 1, 1))
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ground_truth_so_far is not None and len(ground_truth_so_far) > 0:
            ax.legend(loc="upper right")

        fig.tight_layout()

        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        plt.close(fig)
        buf.seek(0)

        plot_img = Image.open(buf).convert("RGB")
        plot_img = plot_img.resize((width, height), Image.Resampling.BILINEAR)
        return np.array(plot_img)

    def save_value_plot(
        self,
        images,
        values,
        filename="value_rollout.mp4",
        fps=10,
        plot_width=500,
        title="Value over time",
        stride=1,
        ground_truth_values=None,
    ):
        ep_len = min(len(images), len(values))
        gt = None
        if ground_truth_values is not None:
            gt = np.asarray(ground_truth_values, dtype=np.float32)
            ep_len = min(ep_len, len(gt))
        images = images[:ep_len]
        values = np.asarray(values[:ep_len], dtype=np.float32)
        if gt is not None:
            gt = gt[:ep_len]
        
        global_min = np.min(values)
        global_max = np.max(values)
        if gt is not None and len(gt) > 0:
            global_min = min(global_min, float(np.min(gt)))
            global_max = max(global_max, float(np.max(gt)))

        if np.isclose(global_min, global_max):
            pad = 1.0
        else:
            pad = 0.1 * (global_max - global_min)
            
        ymin = global_min - pad
        ymax = global_max + pad

        if ep_len == 0:
            raise ValueError("No frames to save.")

        out_path = os.path.join(self.save_dir, filename)

        first_img = self._to_uint8(images[0])
        if first_img.ndim == 2:
            first_img = np.stack([first_img] * 3, axis=-1)
        H, W = first_img.shape[:2]

        with imageio.get_writer(out_path, fps=fps, codec="libx264") as writer:
            for t in range(ep_len):
                img = self._to_uint8(images[t])
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)

                plot_img = self._render_plot(
                    values_so_far=values[: t + 1],
                    ground_truth_so_far=None if gt is None else gt[: t + 1],
                    total_len=ep_len * int(stride),
                    height=H,
                    width=plot_width,
                    ymin=ymin,
                    ymax=ymax,
                    title=title,
                    stride=stride,
                )

                combined = np.concatenate([img, plot_img], axis=1)
                writer.append_data(combined)

        return out_path