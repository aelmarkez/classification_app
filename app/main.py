from pickle import FALSE
from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
import joblib
from PIL import Image, ImageFilter
import json




# Initialize flask app
app = Flask(__name__)



# Handle GET request
@app.route('/', methods=['GET'])
def render():
    return render_template('template.html') 

# Handle POST request
@app.route('/', methods=['POST'])
def post():
    flags = 0
    img = "/" + request.form['filePath']  # Load image
    x=image_prepare(img)
    assert x.all != False,"Openning image probleme : image_prepare()"  

    try:
        # If we use a pipeline structure for our model 
        if flags ==0 :
            model = joblib.load('/models/model.joblib')   # Load prebuilt model
            x = x.reshape(1,-1)
            prediction = model.predict(x)[0]
        # If we use a neural network for our model
        else:
            model = keras.models.load_model('/models/mnist_classification.h5')  # Load prebuilt model
            prediction = np.argmax(model.predict(x.reshape(1, 28, 28, 1)))
        
        assert str(prediction).isdigit()!=False , "Prediction error : model.predict" #test
        assert result_to_json(img,str(prediction)) != False , "File not found : result_to_json()"  #  test
        return render_template('template.html', response=str(prediction))
    except Exception as e:
        return render_template('template.html', response=str(e))



#prepare the image to suit the model (reshpe it to (28,28))

def image_prepare(argv):
    try :
        im = Image.open(argv).convert('L')  ####
    except IOError:
        return FALSE

    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical position
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
    tv = list(newImage.getdata())  # get pixel values
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = np.array(tva) 

    assert tva.shape != (28,28) ,"shape error : image_prepare()"  # Unit test
    return tva


# Create a JSON file contaning the result

def result_to_json(image,result):
    value = { "imagePath" : image,
            "resultat"  : result
    }
    try :
        f = open("/result_json.json", "w")
    except FileNotFoundError:
        return False
    f.write(json.dumps(value))
    return 
