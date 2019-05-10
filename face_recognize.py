from PIL import Image
import face_recognition
import cv2

sample_pic = "images/kalam_sample.jpg"
unknown_pic = "images/unknown2.jpg"

'''load & get encodings of sample image'''
sample_image = face_recognition.load_image_file(sample_pic)
sample_face_encoding = face_recognition.face_encodings(sample_image)[0]

''' recognize all faces from unknown image'''
image = face_recognition.load_image_file(unknown_pic)
#face_locations = face_recognition.face_locations(image)
#uses cnn model #slower and more accurate
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

face_count=0
coordinates = []
'''loop through each face location'''
for face_location in face_locations:
    face_count+=1
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    img_name = "result/"+"Face"+str(face_count)+".jpg"
    pil_image.save(img_name)
    '''compare each face encodings with sample encodings'''
    unknown_face = face_recognition.load_image_file(img_name)
    unknown_face_encoding = face_recognition.face_encodings(unknown_face)[0]
    results = face_recognition.compare_faces([sample_face_encoding], unknown_face_encoding)

    if results[0] == True:
        '''store face locations if the face encodings matches with sample encodings'''
        coordinates.append(face_location)

'''draw bounding boxes using stored face locations'''
img = cv2.imread(unknown_pic)
for coord in coordinates:
    top, right, bottom, left = coord
    cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
cv2.imwrite("result/result.jpg",img)







