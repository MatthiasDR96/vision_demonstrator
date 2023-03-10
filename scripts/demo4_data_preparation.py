import os
import cv2
import torchvision
# from rembg import remove
from torchvision import transforms
from torchvision.io import read_image

input_path = "raw_data2"  # change to the appropriate path
output_path = "processed_data_3"

# Processing steps (comment out unneeded things)
copy = True
crop = True
remove_background = False
resize, dimension = True, 512
transform, variants = False, 3
threshold = False

if __name__ == "__main__":
    for img_class in os.listdir(f"{input_path}"):  # For all classes
        if not os.path.exists(f"{output_path}/{img_class}"):
            os.makedirs(f"{output_path}/{img_class}")

    if copy:
        for img_class in os.listdir(f"{input_path}"):  # For all classes
            print(f"Cropping and/or removing background of {img_class} images if needed and placing them in output folder.")
            for image in os.listdir(f"{input_path}/{img_class}"):
                img = cv2.imread(f"{input_path}/{img_class}/{image}")  # open image
                # Processing steps
                if crop:
                    img = img[0:1080, 840:1920]
                # if remove_background:
                    # img = remove(img)
                cv2.imwrite(f"{output_path}/{img_class}/{image}", img)

    if resize or transform:
        for img_class in os.listdir(f"{output_path}"):  # For all classes
            print(f"Resizing and/or transforming {img_class} images.")
            for image in os.listdir(f"{output_path}/{img_class}"):
                img = read_image(f"{output_path}/{img_class}/{image}")
                if resize:
                    re = transforms.Compose([transforms.Resize([dimension],
                                                               interpolation=transforms.InterpolationMode.BILINEAR)])
                    img = re(img)
                if transform:
                    t1 = transforms.Compose([transforms.RandomAffine(degrees=(5, 355), translate=(0.1, 0.2),
                                                                     scale=(0.7, 1.2))])  # affine transformations
                    t2 = transforms.Compose([transforms.ColorJitter(brightness=[0.8, 1.5], contrast=[0.8, 1.5])])  # color jitter

                    image_original = t2(t1(img))
                    torchvision.utils.save_image(image_original / 255, f"{output_path}/{img_class}/{image}")

                    for amount_of_images in range(variants):
                        copy = t1(t2(img))
                        torchvision.utils.save_image(copy / 255,
                                                f"{output_path}/{img_class}/" + image.strip(".jpg").__str__() +
                                                f"_variant_{amount_of_images}.jpg")
    if threshold:
        for img_class in os.listdir(f"{output_path}"):  # For all classes
            print(f"Thresholding {img_class} images.")
            for image in os.listdir(f"{output_path}/{img_class}"):
                img = cv2.imread(f"{output_path}/{img_class}/{image}")  # open image
                _, img = cv2.threshold(img, 30, 255, cv2.THRESH_TOZERO)
                cv2.imwrite(f"{output_path}/{img_class}/{image}", img)

    print("Done.")
