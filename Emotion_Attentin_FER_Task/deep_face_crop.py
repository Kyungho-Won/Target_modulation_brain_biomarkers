from deepface import DeepFace
import os
import cv2
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

dir_path = "/home/ubuntu/ksy/SNU/dataset/total_images"
save_dir = "/home/ubuntu/ksy/SNU/dataset/deep_face_crop"
save_dir2 = "/home/ubuntu/ksy/SNU/dataset/deepface_crop_fail"
image_dir = os.listdir(dir_path)
f = open('/home/ubuntu/ksy/SNU/dataset/deepface_crop_fail/crop_fail_list.txt', 'w')
for image_name in image_dir:
    if os.path.exists(os.path.join(save_dir, image_name)):
        continue
    try:
        image = DeepFace.detectFace(img_path=os.path.join(
            dir_path, image_name),  detector_backend=backends[4],align=False)
        image = image*255

        print(image_name)

        cv2.imwrite(os.path.join(save_dir, image_name), 
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    except:
        print("face not cropped: {}".format(image_name))
        f.write(image_name)
        f.write('\n')
        # cv2.imwrite(os.path.join(save_dir2, image_name), os.path.join(dir_path, image_name))
                    # cv2.cvtColor(os.path.join(dir_path, image_name), cv2.COLOR_RGB2BGR))
        continue

