# data/transforms.py
import numpy as np
from PIL import Image


class BlockPixelShuffle:
    def __init__(self, block_size=4):
        """
        初始化块大小。

        :param block_size: 每个块的大小（默认为4，即4x4像素）
        """
        self.block_size = block_size

    def __call__(self, img):
        """
        对图像进行块内像素洗牌。

        :param img: PIL Image 对象
        :return: 处理后的 PIL Image 对象
        """
        img_array = np.array(img)

        # 检查图像是单通道还是多通道
        if img_array.ndim == 2:
            # 单通道图像
            channels = 1
            img_array = img_array[:, :, np.newaxis]
        else:
            # 多通道图像（例如RGB）
            channels = img_array.shape[2]

        height, width, _ = img_array.shape

        # 计算在x和y方向上的块数
        blocks_x = width // self.block_size
        blocks_y = height // self.block_size

        # 遍历每个块
        for by in range(blocks_y):
            for bx in range(blocks_x):
                # 定位当前块的起始坐标
                start_x = bx * self.block_size
                start_y = by * self.block_size

                # 提取当前块
                block = img_array[start_y:start_y + self.block_size, start_x:start_x + self.block_size, :]

                # 将块展平成一维数组
                pixels = block.reshape(-1, channels)

                # 打乱像素顺序
                np.random.shuffle(pixels)

                # 将打乱后的像素重新赋值回块
                img_array[start_y:start_y + self.block_size, start_x:start_x + self.block_size, :] = pixels.reshape(
                    self.block_size, self.block_size, channels)

        # 如果是单通道图像，去掉最后一个维度
        if channels == 1:
            img_array = img_array.squeeze(axis=2)

        # 将处理后的数组转换回 PIL Image
        shuffled_img = Image.fromarray(img_array)
        return shuffled_img