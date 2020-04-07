# object_detection


This script takes in a folder of images and uses Coral AI detection model mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite to "detect" objects within the images.
Based on the script selected confidence % boxes are drawn around the image object and the resulting .jpg
is stored in the /object folder. So if a person was detected and was over the confidence % that person with a box willl be stored in the parameter supplied folder in subfolder /person.  This done for each supplied target object. For examples, cars in /car subfolder, dogs in /dog subfolder and so on.

The arguments for this script are:<br>
---host          :Host IP of the coral ai server<br>
--port           :Port of the coral ai server<br>
--folder         :file folder for processing<br>
--targets        :Objects list<br>
--confidence     :Confidence of target to be detected<br>
--save_folder    :Folder where detected objects are to be placed<br>
--timestamp      :Files to be timestamped T or F<br>

The Coral AI server is setup on HOST and port using https://github.com/robmarkcole/coral-pi-rest-server . <br>
This exposes Tensorflow-lite models on a Coral usb accelerator via a Flask app.


