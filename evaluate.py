"""
图像质量评估脚本 (兼容Kornia 0.5+)
功能：
1. 批量计算生成图像与真实图像的PSNR(基于skimage)
2. 使用Kornia计算SSIM (自动兼容新旧版本)
3. 自动处理灰度/彩色图像
4. 动态调整窗口尺寸
"""

from skimage.metrics import peak_signal_noise_ratio as psnr
import kornia
import torch
import numpy as np
from PIL import Image
import os


def load_image_pair(fake_path, real_path):
    """加载图像对并确保一致性"""

    # 加载图像
    def _load(path):
        img = np.array(Image.open(path))
        if img.ndim == 2:  # 灰度图扩展通道维度
            img = np.expand_dims(img, axis=-1)
        return img.astype(np.float32)

    fake = _load(fake_path)
    real = _load(real_path)

    # 尺寸校验
    if fake.shape != real.shape:
        raise ValueError(f"尺寸不匹配: {fake.shape} vs {real.shape}")

    return fake, real


def to_kornia_tensor(arr):
    """转换到与训练一致的[-1,1]范围"""
    tensor = torch.from_numpy(arr).float()
    tensor = (tensor / 127.5) - 1.0  # [0,255] -> [-1,1]
    return tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW


def calculate_ssim(fake_tensor, real_tensor):
    """兼容不同Kornia版本的SSIM计算"""
    # 动态窗口调整
    h, w = fake_tensor.shape[-2:]
    win_size = min(11, h, w)  # 与训练时最大窗口11一致
    win_size = win_size if win_size % 2 == 1 else win_size - 1

    # 版本兼容处理
    kwargs = {
        'window_size': win_size,
        'max_val': 2.0  # 对应[-1,1]范围
    }
    if hasattr(kornia.metrics, 'ssim'):  # 新版kornia
        ssim_val = kornia.metrics.ssim(fake_tensor, real_tensor, **kwargs)
    else:  # 旧版kornia (<0.6)
        ssim_val = kornia.losses.ssim(fake_tensor, real_tensor, **kwargs)

    return ssim_val.mean().item()


def calculate_metrics(fake_path, real_path):
    """核心指标计算函数"""
    # 加载并校验图像对
    fake_arr, real_arr = load_image_pair(fake_path, real_path)

    # 计算PSNR (使用原始范围[0,255])
    psnr_val = psnr(real_arr, fake_arr, data_range=255)

    # 转换到PyTorch Tensor (自动处理通道顺序)
    fake_tensor = to_kornia_tensor(fake_arr)
    real_tensor = to_kornia_tensor(real_arr)

    # 计算SSIM
    with torch.no_grad():
        ssim_val = calculate_ssim(fake_tensor, real_tensor)

    return psnr_val, ssim_val


def batch_process(image_dir):
    """批量处理目录中的图像"""
    psnr_values = []
    ssim_values = []

    # 自动检测所有_fake图像
    fakes = [f for f in os.listdir(image_dir)
             if "_fake" in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    print(f"发现 {len(fakes)} 张生成图像")

    for fake_name in fakes:
        real_name = fake_name.replace("_fake_B", "_real_B")
        fake_path = os.path.join(image_dir, fake_name)
        real_path = os.path.join(image_dir, real_name)

        if not os.path.exists(real_path):
            print(f"⚠️ 缺少配对图像: {real_name}")
            continue

        try:
            psnr_val, ssim_val = calculate_metrics(fake_path, real_path)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            print(f"✅ {fake_name} -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        except Exception as e:
            print(f"❌ 处理失败 {fake_name}: {str(e)}")

    # 结果统计
    if psnr_values:
        print("\n📊 汇总结果:")
        print(f"平均PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
        print(f"平均SSIM: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
        print(f"有效样本: {len(psnr_values)} 对")
    else:
        print("\n⚠️ 未找到有效图像对")


if __name__ == "__main__":
    # 配置参数
    TEST_IMAGE_DIR = "./results//test_latest/images"  # 修改为你的测试目录

    # 执行评估
    print("🚀 开始图像质量评估...")
    batch_process(TEST_IMAGE_DIR)
    print("✅ 评估完成")