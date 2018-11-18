import cv2
import time
import pytesseract
from PIL import ImageGrab
import numpy as np
from collections import Counter

MINIMAP_COORDS = [1626, 14, 1905, 293]
# COMPASS_COORDS = [942, 3, 978, 19]
HEALTH_COORDS = [771, 965, 808, 986]
ARMOUR_COORDS = [771, 937, 808, 958]
MINIMAP_SIZE = [280, 280]
TEST_SIZE = (255, 255)
MAP_FILE = "./map.png"
MAP_GRAY = cv2.imread(MAP_FILE, 0)
MAP_COLOR = cv2.imread(MAP_FILE, 1)

def grab_screenshot(coords):
    original_img = ImageGrab.grab(bbox=coords)
    original_array = np.array(original_img, dtype="uint8")
    grayscaled_img = cv2.cvtColor(original_array, cv2.COLOR_BGR2GRAY)
    return grayscaled_img

def find_location(minimap):
    img = cv2.resize(minimap, TEST_SIZE)
    votes = Counter()

    template = MAP_GRAY.copy()
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)

    _, _, top_left, _ = cv2.minMaxLoc(res)
    res_x = int(top_left[0] + TEST_SIZE[0] / 2)
    res_y = int(top_left[1] + TEST_SIZE[1] / 2)
    return (res_x, res_y)

def print_coords(coords):
    print("({}, {})".format(coords[0], coords[1]))
    return

def draw_coords(coords):
    img = MAP_COLOR.copy()
    cv2.circle(img, coords, 10, (0, 0, 255), thickness=-1)
    cv2.circle(img, coords, 40, (0, 0, 255), thickness=2)
    img_resized = cv2.resize(img, (500, 500))
    cv2.imshow("Map", img_resized)
    cv2.waitKey(1)
    return

def find_health(health_img):
    health_img = cv2.resize(health_img, None, fx=2.0, fy=2.0,
                            interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1, 1), np.uint8)
    health_img = cv2.threshold(health_img, 180, 255,
                               cv2.THRESH_BINARY_INV)[1]
    health_str = pytesseract.image_to_string(health_img, lang="fortnite",
                                             config="outputbase digits")
    if health_str.isdigit():
        return int(health_str)
    else:
        return None

def print_health(health):
    if health is not None:
        print("Health:", health)
    else:
        print("Health not found.")
    return

def find_armour(armour_img):
    armour_img = cv2.resize(armour_img, None, fx=2.0, fy=2.0,
                            interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1, 1), np.uint8)
    armour_img = cv2.threshold(armour_img, 180, 255,
                               cv2.THRESH_BINARY_INV)[1]
    armour_str = pytesseract.image_to_string(armour_img, lang="fortnite",
                                             config="outputbase digits")
    if armour_str.isdigit():
        return int(armour_str)
    else:
        return None

def print_armour(armour):
    if armour is not None:
        print("Armour:", armour)
    else:
        print("Armour not found.")
    return

def main():
    print("Map filepath:", MAP_FILE)
    print("Test size: ({}, {})".format(TEST_SIZE[0], TEST_SIZE[1]))
    while True:
        coords = find_location(grab_screenshot(MINIMAP_COORDS))
        print_coords(coords)
        draw_coords(coords)
        health = find_health(grab_screenshot(HEALTH_COORDS))
        print_health(health)
        armour = find_armour(grab_screenshot(ARMOUR_COORDS))
        print_armour(armour)

    return

if __name__ == "__main__":
    main()
