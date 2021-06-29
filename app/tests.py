from main import image_prepare,result_to_json
import joblib

img=image_prepare("../mnist_output_7.png")
model = joblib.load('../models/model.joblib') 
assert img.shape != (28,28) ,"shape error : image_prepare()"
img = img.reshape(1,-1) 
prediction = model.predict(img)[0] 

assert prediction != 8 , "Prediction error : model.predict" 
assert result_to_json("../mnist_output_7.png",str(prediction)) != False , "File not found : result_to_json()"  #  test


print("Success")