"""
Component that will perform object detection and identification via coral AI. Places green boxes around objects
that are within the selected confidence % and slots them into the "object" sub-directory. Will delete source images after completion.
Uses flask based coral ai service to provide predications. See https://github.com/robmarkcole/coral-pi-rest-server
version 1.0 April 3,2020

"""
import base64
import datetime
import io
import json
import os
import sys
import argparse
from typing import Tuple
import requests
from PIL import Image, ImageDraw
import glob



DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
BOX = "box"
RED = (255, 0, 0)
GREEN = (0, 128, 0)
HTTP_OK = 200


class ObjectClassifyEntity:
    """Perform a object classification."""

    def __init__(
        self,
        ip_address,
        port,
        image,
        target,
        confidence,
        save_file_folder,
        save_timestamped_file,
 
    ):
        timeout = 10
        api_key = ''
        self.image = image
        self._ip_address = ip_address
        self._port = port
        self._api_key = ''
        self._timeout = timeout
        self._target = target
        self._confidence = confidence
        self._name = 'google_cams'
        self._state = None
        self._targets_confidences = [None] * len(self._target)
        self._targets_found = [0] * len(self._target)
        self._predictions = {}
        self._summary = {}
        self._last_detection = None
        self._image_width = None
        self._image_height = None
        self._image = image
        if save_file_folder:
            self._save_file_folder = save_file_folder
        self._save_timestamped_file = save_timestamped_file

    def generate(self):
        
        """Run the process on an image"""
        
        self.process_image()
        
    def process_image(self):
        """Process an image."""
        print(self._image) 
        self._image_width, self._image_height = Image.open(self._image).size
        print('Size ',self._image_width, self._image_height) 
        self._state = None
        self._targets_confidences = [None] * len(self._target)
        self._targets_found = [0] * len(self._target)
        self._predictions = {}
        self._summary = {}
        self.detect()

        print('Predictions', self._predictions)
        if self._predictions:
            for i, target in enumerate(self._target):

                raw_confidences = [pred["confidence"] for pred in self._predictions if pred["label"] == target]

                self._targets_confidences[i] = [round(float(confidence) * 100, 1) for confidence in raw_confidences]

                print( self._targets_confidences[i])
                self._targets_found[i] = len([val for val in self._targets_confidences[i] if val >= self._confidence])
                print( self._targets_found[i])
            self._state = sum(self._targets_found)
            if self._state > 0:
                self._last_detection = datetime.datetime.now().strftime(DATETIME_FORMAT)

            labels = [pred["label"] for pred in self._predictions]
            gobjects = list(set(labels))
            tconfidences = len([pred["confidence"] for pred in self._predictions if pred["label"] == target])
            
            self._summary = {target: tconfidences for target in gobjects}
            main_date = datetime.datetime.now()

            if self._state > 0:
                self.save_image(
                    self._image, self._predictions, self._target, self._save_file_folder, main_date
                )

    def save_image(self, image, predictions, target, directory, main_date):
        """Save a timestamped image with bounding boxes around targets."""

        img= Image.open(self._image ,'r')
        draw = ImageDraw.Draw(img)

        for prediction in predictions:

            prediction_confidence = round(float(prediction["confidence"]) * 100, 1)
            if (
                prediction["label"] in target
                and prediction_confidence >= self._confidence
            ):
                box = self.get_box(prediction, self._image_width, self._image_height)
                print('Red box ', box)
                self.draw_box(
                    draw,
                    box,
                    self._image_width,
                    self._image_height,
                    text=str(prediction_confidence),
                    color=RED,
                )

        latest_save_path = directory + "{}_latest_all.jpg".format(self._name )
        img.save(latest_save_path)

        if self._save_timestamped_file:
            date = main_date
            timestamp_save_path = directory + "{0}_all_{1:%Y%m%d}_{2:%H%M%S}.jpg".format(self._name, date, date ) 

            out_file = open(timestamp_save_path, "wb")
            img.save(out_file, format="JPEG")
            out_file.flush()
            os.fsync(out_file)
            out_file.close()

            print("Saved bounding box image to %s", timestamp_save_path)
        for singletarget in target:
            flag_target_found = False
            targetimg = Image.open(self._image ,'r')
            draw = ImageDraw.Draw(targetimg) 
            for prediction in predictions:

                prediction_confidence = round(float(prediction["confidence"]) * 100, 1)
                if (
                    prediction["label"] == singletarget
                    and prediction_confidence >= self._confidence
                ):
                    flag_target_found = True
                    box = self.get_box(prediction, self._image_width, self._image_height)
                    print('Green ' , box)
                    self.draw_box(
                        draw,
                        box,
                        self._image_width,
                        self._image_height,
                        text=str(prediction_confidence),
                        color=GREEN,
                    )
            if flag_target_found:       
               date = main_date
               latest_save_path = directory +  "{0}/{1}_{2}_{3:%Y%m%d}_{4:%H%M%S}.jpg".format(singletarget, self._name, singletarget, date, date)
               targetimg.save(latest_save_path) 
           
    def get_box(self,prediction, img_width, img_height):
        """
        Return the relative bounxing box coordinates.

        Defined by the tuple (y_min, x_min, y_max, x_max)
        where the coordinates are floats in the range [0.0, 1.0] and
        relative to the width and height of the image.
        """
        print( 'H x W ', img_height, ' ', img_width)
        print( 'Pred min and maxs ', prediction["y_min"],  prediction["x_min"], prediction["y_max"], prediction["x_max"])
        box = [
            float(prediction["y_min"]) / float(img_height),
            float(prediction["x_min"]) / float(img_width),
            float(prediction["y_max"]) / float(img_height),
            float(prediction["x_max"]) / float(img_width),
        ]
        
        rounding_decimals = 3
        box = [round(coord, rounding_decimals) for coord in box]
        print(' Get box ', box)
        return box                    

    def detect(self):
        """Process image_bytes, performing detection."""
        self._predictions = []
        url = "http://{}:{}/v1/vision/detection".format(self._ip_address, self._port)
        print(url)
       
        files = {'image': open(self._image, 'r')}
        print(files)
        try:
            response = requests.post(url, files=files)
            print('Response ', response)

        except:
            print('Post error to call')


        if response.status_code == HTTP_OK:
            if response.json()["success"]:
                self._predictions = response.json()["predictions"]
            else:
                print('Engine pooped out..no predictions made')
                self._predictions = ''

    def draw_box(self, draw, box, img_width, img_height, text, color): 
        """
        Draw a bounding box on and image.
        The bounding box is defined by the tuple (y_min, x_min, y_max, x_max)
        where the coordinates are floats in the range [0.0, 1.0] and
        relative to the width and height of the image.
        For example, if an image is 100 x 200 pixels (height x width) and the bounding
        box is `(0.1, 0.2, 0.5, 0.9)`, the upper-left and bottom-right coordinates of
        the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).
        """

        line_width = 3
        font_height = 8
        y_min, x_min, y_max, x_max = box
        (left, right, top, bottom) = (
            x_min * img_width,
            x_max * img_width,
            y_min * img_height,
            y_max * img_height,
        )
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
            width=line_width,
            fill=color,
        )
        if text:
            draw.text(
                (left + line_width, abs(top - line_width - font_height)), text, fill=color
            )


def main():

    parser = argparse.ArgumentParser(description='This script takes a image jpg file and based on confidence draws boxes around identified targets.')
    parser.add_argument('-w','--host', help='Host IP of the coral ai server',required=True)
    parser.add_argument('-p','--port', help='Port of the coral ai server', required=True)
    parser.add_argument('-f','--folder', help='file folder for processing', required=True)
    parser.add_argument('-t','--targets',  help='Objects list', required=True)
    parser.add_argument('-c','--confidence', type=int, help='Confidence of target to be detected', required=True)
    parser.add_argument('-s','--save_folder', help='Folder where detected objects are to be placed', required=True)
    parser.add_argument('-y','--timestamp', help='Files to be timestamped T or F', required=True)
    args = parser.parse_args()

    ip_address            = args.host
    port                  = args.port 
    process_folder        = args.folder
    target                = args.targets
    confidence            = args.confidence
    save_file_folder      = args.save_folder
    list_of_images        = []
    if save_file_folder:
        save_file_folder = os.path.join(save_file_folder, "")  # If no trailing / add it
    save_timestamped_file = args.timestamp
    list_of_images = os.listdir(process_folder)
    print(save_file_folder)
    for i, ind in enumerate(list_of_images):
        list_of_images[i] = str(process_folder + '/' + ind)

    print('Processing ', len(list_of_images), ' files ', list_of_images)
    target = list(target.split(","))
    for obj in target:
        if not os.path.exists(save_file_folder +  obj):
           os.mkdir(save_file_folder + obj)
           print('Folder created: ', save_file_folder + obj)
#    sys.exit()
    for ind_file_image in list_of_images:
        print('Processing image: ', ind_file_image)
        CoralGenerator = ObjectClassifyEntity(ip_address, port, ind_file_image, target, confidence, save_file_folder, save_timestamped_file)
        CoralGenerator.generate()
# finally remove files from the incoming folder
    all_files = glob.glob(process_folder + '/*.*')
    files_removed = 0
    for f in all_files:
        os.remove(f)
        print('Removed file ', f)
        files_removed = files_removed + 1
    print('Total of ', files_removed, ' were removed..done')
    

if __name__ == "__main__":
    main()

