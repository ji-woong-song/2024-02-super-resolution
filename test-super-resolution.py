
import torch
from PIL import Image
import os
from torchvision import transforms
from superResolution import SRCNN
from tqdm import tqdm

def process_images(model_path, input_dir, output_dir, device='cuda'):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로드
    model = SRCNN()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 변환을 위한 transform 정의
    transform = transforms.Compose([
        transforms.Resize((570, 564), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

    # 이미지 파일 리스트
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = [f for f in os.listdir(input_dir) 
             if os.path.splitext(f)[1].lower() in valid_extensions]

    print("이미지 초해상도 처리 중...")
    with torch.no_grad():
        for filename in tqdm(files):
            try:
                # 이미지 로드 및 전처리
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                # 초해상도 처리
                output = model(img_tensor)
                
                # 텐서를 이미지로 변환
                output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
                
                # 결과 저장
                output_path = os.path.join(output_dir, filename)
                output_img.save(output_path, quality=95)

            except Exception as e:
                print(f"\n{filename} 처리 중 오류: {str(e)}")

    print("\n처리 완료!")

if __name__ == "__main__":
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 경로 설정
    model_path = '../super-resolution/srcnn_best_model.pth'
    input_dir = '../super-resolution/lr'
    output_dir = 'result'
    
    # 이미지 처리 실행
    process_images(model_path, input_dir, output_dir, device)
