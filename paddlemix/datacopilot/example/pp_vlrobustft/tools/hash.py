import scipy
import paddle
import numpy as np
from PIL import Image
from typing import List, Tuple

class ImageDeduplicator:
    def __init__(self, hash_size=8):
        self.hash_size = hash_size

    def compute_dhash(self, img) -> np.ndarray:
        """
        计算dHash值，返回布尔数组
        返回: numpy.ndarray shape=(hash_size, hash_size-1)
        """
        img = img.convert('L').resize((self.hash_size + 1, self.hash_size))
        pixels = np.array(img)
        return pixels[:, 1:] > pixels[:, :-1]

    def compute_phash(self, img) -> np.ndarray:
        """
        计算pHash值，返回布尔数组
        返回: numpy.ndarray shape=(hash_size, hash_size)
        """
        img = img.convert('L').resize((32, 32))
        pixels = np.array(img)
        dct = scipy.fft.dct(scipy.fft.dct(pixels, axis=0), axis=1)
        dct_low = dct[:self.hash_size, :self.hash_size]
        return dct_low > dct_low.mean()

    def hamming_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> int:
        """计算两个哈希值的汉明距离"""
        return np.count_nonzero(hash1 != hash2)

    def are_similar(self, hash1: np.ndarray, hash2: np.ndarray, threshold: int = 5) -> bool:
        """判断两个哈希值是否相似"""
        return self.hamming_distance(hash1, hash2) <= threshold

    def find_duplicates(self, images: List[Image.Image]) -> List[Tuple[int, int, int]]:
        """
        查找重复图片
        返回: List[Tuple[idx1, idx2, distance]]
        """
        # 计算所有图片的哈希值
        hashes = [self.compute_dhash(img) for img in images]
        
        # 查找相似对
        duplicates = []
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                distance = self.hamming_distance(hashes[i], hashes[j])
                if distance <= self.threshold:
                    duplicates.append((i, j, distance))
        
        return duplicates