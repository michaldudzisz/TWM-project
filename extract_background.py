import cv2
import numpy as np
import glob
import os


def resize_image(image):
    target_width = 400
    target_height = 400
    return cv2.resize(image, (target_width, target_height))


digits = [chr(i) for i in range(ord('0'), ord('9') + 1)]
letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
labels = digits + letters

if __name__ == "__main__":

    raw_photos_path = "owndataset/raw_photos/"
    for label in labels:
        raw_files_direcotry = raw_photos_path + label
        for file_name in os.listdir(raw_files_direcotry):

            file_path = os.path.join(raw_files_direcotry, file_name)

            image = cv2.imread(file_path)
            image = resize_image(image)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_dark_green1 = np.array([50, 0, 5]) # background is about: hsv(177, 69%, 34%)  np.array([50, 0, 5])
            upper_dark_green1 = np.array([330, 255, 200]) # np.array([110, 255, 230])

            lower_dark_green2 = np.array([0, 0, 10]) 
            upper_dark_green2 = np.array([10, 0, 100]) 

            mask1 = cv2.inRange(hsv, lower_dark_green1, upper_dark_green1)
            mask2 = cv2.inRange(hsv, lower_dark_green2, upper_dark_green2)
            # mask = cv2.bitwise_and(cv2.bitwise_not(mask1), cv2.bitwise_not(mask2))

            mask1 = cv2.bitwise_not(mask1)
            mask2 = cv2.bitwise_not(mask2)

            # Mask inversion - white hand, black background
            # mask = cv2.bitwise_not(mask)
            # mask = mask2
            mask = cv2.bitwise_and(mask1, mask2)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=0) # Dylatacja - rozszerza białe obszary
            mask = cv2.erode(mask, kernel, iterations=2) # Erozja - usuwa szumy
            mask = cv2.GaussianBlur(mask, (3, 3), 0.4) # Rozmycie maski, aby uzyskać gładsze krawędzie
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Zamknięcie - wypełnia małe dziury
            # mask = cv2.GaussianBlur(mask, (3, 3), 100) # Rozmycie maski, aby uzyskać gładsze krawędzie
            hand = cv2.bitwise_and(image, image, mask=mask)

            background = np.zeros_like(image)

            # Add hand to black background
            result = cv2.add(hand, background)

            try: 
                new_directory = './owndataset/processed/' + label
                print(new_directory)
                os.mkdir(new_directory)
            except:
                pass # directory already exists

            cv2.imwrite('./owndataset/processed/' + label + '/' + file_name, result)






