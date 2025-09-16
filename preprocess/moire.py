import cv2
import numpy as np
import random


class Moire:

    def __init__(self):
        pass

    def add_moire_noise(self, src):
        height, width = src.shape[:2]
        center = (height // 2, width // 2)
        degree = random.uniform(0.0005, 0.01)

        # Tạo tọa độ lưới
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Tính toán độ lệch từ tâm
        offset_X = X - center[0]
        offset_Y = Y - center[1]

        # Tính góc và bán kính trong tọa độ cực
        theta = np.arctan2(offset_Y, offset_X)
        rou = np.sqrt(offset_X ** 2 + offset_Y ** 2)

        # Tính toán tọa độ mới
        new_X = center[0] + rou * np.cos(theta + degree * rou)
        new_Y = center[1] + rou * np.sin(theta + degree * rou)

        # Phạm vi tọa độ giới hạn
        new_X = np.clip(new_X, 0, width - 1).astype(np.int32)
        new_Y = np.clip(new_Y, 0, height - 1).astype(np.int32)

        # Áp dụng hiệu ứng moiré
        dst = 0.8 * src + 0.2 * src[new_Y, new_X]

        return dst.astype(np.uint8)

    def __call__(self, img):
        img = self.add_moire_noise(img)
        return img


# if __name__ == "__main__":
#     img = cv2.imread('005700.jpg')
#     moire = Moire()
#     img = moire(img)
#     cv2.imwrite('transforms_moire.png', img)