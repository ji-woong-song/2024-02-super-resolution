import os
from PIL import Image
from tqdm import tqdm


def find_minimum_resolution(input_dir):
    """
    디렉토리 내 모든 이미지의 최소 해상도를 찾습니다.
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    min_width = float('inf')
    min_height = float('inf')

    # 이미지 파일 리스트
    files = [f for f in os.listdir(input_dir)
             if os.path.splitext(f)[1].lower() in valid_extensions]

    print("최소 해상도 탐색 중...")
    for filename in tqdm(files):
        try:
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                min_width = min(min_width, width)
                min_height = min(min_height, height)
        except Exception as e:
            print(f"\n{filename} 처리 중 오류: {str(e)}")

    return min_width, min_height


def center_crop_to_minimum(input_dir, output_dir, target_width, target_height):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 지원하는 이미지 확장자
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # 이미지 파일 리스트
    files = [f for f in os.listdir(input_dir)
             if os.path.splitext(f)[1].lower() in valid_extensions]

    print("\n이미지 크롭 처리 중...")
    skipped = []
    for filename in tqdm(files):
        try:
            # 이미지 로드
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # RGBA 이미지를 RGB로 변환
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # 현재 이미지 크기
            width, height = img.size

            # 중앙 크롭을 위한 좌표 계산
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            right = left + target_width
            bottom = top + target_height

            # 이미지 크롭
            cropped_img = img.crop((left, top, right, bottom))

            # 크롭된 이미지 저장
            output_path = os.path.join(output_dir, filename)
            cropped_img.save(output_path, quality=95)

        except Exception as e:
            print(f"\n{filename} 처리 중 오류: {str(e)}")
            skipped.append((filename, str(e)))

    # 처리 결과 출력
    print("\n처리 완료!")
    print(f"모든 이미지가 {target_width} x {target_height} 해상도로 조정되었습니다.")

    if skipped:
        print("\n처리하지 못한 이미지들:")
        for filename, reason in skipped:
            print(f"- {filename}: {reason}")

    return target_width, target_height


def left_top_crop_to_minimum(input_dir, output_dir, target_width, target_height):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 지원하는 이미지 확장자
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    # 이미지 파일 리스트
    files = [f for f in os.listdir(input_dir)
             if os.path.splitext(f)[1].lower() in valid_extensions]

    print("\n이미지 크롭 처리 중...")
    skipped = []
    for filename in tqdm(files):
        try:
            # 이미지 로드
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # RGBA 이미지를 RGB로 변환
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # 현재 이미지 크기
            width, height = img.size

            # 중앙 크롭을 위한 좌표 계산
            left = 0
            top = 0
            right = left + target_width
            bottom = top + target_height

            # 이미지 크롭
            cropped_img = img.crop((left, top, right, bottom))

            # 크롭된 이미지 저장
            output_path = os.path.join(output_dir, filename)
            cropped_img.save(output_path, quality=95)

        except Exception as e:
            print(f"\n{filename} 처리 중 오류: {str(e)}")
            skipped.append((filename, str(e)))

    # 처리 결과 출력
    print("\n처리 완료!")
    print(f"모든 이미지가 {target_width} x {target_height} 해상도로 조정되었습니다.")

    if skipped:
        print("\n처리하지 못한 이미지들:")
        for filename, reason in skipped:
            print(f"- {filename}: {reason}")

    return target_width, target_height