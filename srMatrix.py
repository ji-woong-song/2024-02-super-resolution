import os
from PIL import Image
import numpy as np
from sewar.full_ref import psnr, ssim, msssim
import pandas as pd
from torchvision import transforms
import torch
from pytorch_msssim import ms_ssim


def calculate_average_metrics(result_dir, hr_dir):
    """
    결과 이미지와 원본 고해상도 이미지의 품질 평가 메트릭의 평균을 계산
    
    Args:
        result_dir (str): Super Resolution 결과 이미지가 있는 디렉토리
        hr_dir (str): 원본 고해상도 이미지가 있는 디렉토리
    """
    # 결과를 저장할 리스트들
    psnr_values = []
    ssim_values = []
    ifc_values = []
    nqm_values = []
    wpsnr_values = []
    msssim_values = []
    
    # 결과 디렉토리의 모든 이미지 파일
    result_files = os.listdir(result_dir)
    
    for img_name in result_files:
        # 동일한 이름의 HR 이미지가 있는지 확인
        hr_path = os.path.join(hr_dir, img_name)
        if not os.path.exists(hr_path):
            print(f"Warning: No matching HR image for {img_name}")
            continue
            
        # 이미지 로드
        result_img = Image.open(os.path.join(result_dir, img_name))
        hr_img = Image.open(hr_path)
        
        # numpy 배열로 변환
        result_array = np.array(result_img)
        hr_array = np.array(hr_img)
        
        # MS-SSIM 계산을 위한 torch tensor 변환
        to_tensor = transforms.ToTensor()
        result_tensor = to_tensor(result_img).unsqueeze(0)
        hr_tensor = to_tensor(hr_img).unsqueeze(0)
        
        # 메트릭 계산
        psnr_values.append(psnr(result_array, hr_array))
        ssim_values.append(ssim(result_array, hr_array)[0])
        # ifc_values.append(ifc(result_array, hr_array))
        # nqm_values.append(nqm(result_array, hr_array))
        # wpsnr_values.append(wpsnr(result_array, hr_array))
        msssim_values.append(ms_ssim(result_tensor, hr_tensor, data_range=1.0).item())
        
        print(f"Processed {img_name}")
    
    # 평균 계산
    avg_metrics = {
        'PSNR': np.mean(psnr_values),
        'SSIM': np.mean(ssim_values),
        'MS-SSIM': np.mean(msssim_values)
    }
    
    # 결과 출력
    print("\nAverage Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return avg_metrics

# 사용 예시
if __name__ == "__main__":
    result_dir = "result"  # Super Resolution 결과가 있는 폴더
    hr_dir = "hr"         # 원본 고해상도 이미지가 있는 폴더
    
    avg_metrics = calculate_average_metrics(result_dir, hr_dir)