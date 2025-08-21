import cv2


def print_image_information(image):
    height, width, channel = image.shape
    print("Height: ", height)
    print("Width: ", width)
    print("Channel", channel)
    print("File Type:", image.dtype)
    print("Size: ", image.size)

def camera(cam):

    # Get FPS, width, and height
    fps = cam.get(cv2.CAP_PROP_FPS)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Release camera
    cam.release()

    # Save to a text file
    with open("solutions/camera_info.txt", "w") as f:
        f.write(f"FPS: {fps}\n")
        f.write(f"Width: {int(width)}\n")
        f.write(f"Height: {int(height)}\n")

    print("Camera info saved to camera_info.txt")


def main():
    image = cv2.imread("lena-1.png")
    cam = cv2.VideoCapture(0)


    print_image_information(image)
    camera(cam)


if __name__ == "__main__":
    main()