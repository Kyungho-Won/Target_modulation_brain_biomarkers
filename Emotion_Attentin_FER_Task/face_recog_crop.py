import cv2
import face_recognition
import os
dir_path = "/home/ubuntu/ksy/SNU/dataset/concat"
save_dir = "/home/ubuntu/ksy/SNU/dataset/face_crop1"
save_dir2 = "/home/ubuntu/ksy/SNU/dataset/not_face"
image_dir = os.listdir(dir_path)
for image_name in image_dir:
    if os.path.exists(os.path.join(save_dir, image_name)):
        continue
    image = face_recognition.load_image_file(
        os.path.join(dir_path, image_name))
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        print("face not detected")
        cv2.imwrite(os.path.join(save_dir2, image_name),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        continue
    try:
        print(image_name)
        bottom, right, top, left = face_locations[0]
        cropped_img = image[bottom-20:top+20, left-20:right+20]   
        cv2.imwrite(os.path.join(save_dir,image_name),
                cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        # print("saved")
    except:
        print("face not cropped: {}".format(image_name))
        cv2.imwrite(os.path.join(save_dir2, image_name),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        continue
