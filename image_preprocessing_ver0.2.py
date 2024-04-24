from PIL import Image
import numpy as np
import glob

paths = glob.glob("E:MachineLearning\\discriminator\\gender\\F\\*\\*.jpg")

# 모든 이미지를 동일한 크기로 조정할 새로운 크기 지정
new_size = (200, 200)

# 동일한 크기로 조정된 이미지를 담을 리스트 생성
resized_images = []

for path in paths:
    # 이미지 열기
    img = Image.open(path)
    
    # 이미지 크기 조정
    resized_img = img.resize(new_size, Image.ANTIALIAS)  # 크기 조정 알고리즘 선택
    
    # RGB 채널 조정 (옵션)
    # resized_img = resized_img.convert('RGB')  # 예시: RGB 채널로 변환
    
    # 이미지를 numpy 배열로 변환하여 리스트에 추가
    resized_images.append(np.array(resized_img))

# 리스트를 numpy 배열로 변환
resized_images_array = np.array(resized_images)
