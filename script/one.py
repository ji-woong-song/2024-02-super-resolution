import process.resolution as res

# 사용 예시
if __name__ == "__main__":
    input_dir = "../hr_origin"  # 입력 디렉토리
    output_dir = "../hr"  # 출력 디렉토리

    """
    모든 이미지를 최소 해상도로 center-crop 합니다.
    """
    # 최소 해상도 찾기
    target_width, target_height = res.find_minimum_resolution(input_dir)
    print(f"\n발견된 최소 해상도: {target_width} x {target_height}")

    # 이미지 처리 실행
    final_width, final_height = res.center_crop_to_minimum(input_dir, output_dir, target_width, target_height)

    # 결과 출력
    print(f"\n최종 해상도: {final_width} x {final_height}")