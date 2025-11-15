import cv2

def convert_jpg_to_grayscale_png_cv2(input_path, output_path):
    try:
        # Read the image in color mode (cv2 reads as BGR by default)
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Could not read the image from {input_path}")
            return

        # Convert the BGR image to grayscale
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image as a PNG file
        cv2.imwrite(output_path, grayscale_img)
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


convert_jpg_to_grayscale_png_cv2("./globalProfilePic_LW.jpg", "../images/globalProfilePic_LW_GS.png")