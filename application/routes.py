from flask import render_template
from flask import Response
from flask import request

from application import app
from application.Camera import Camera
from application.forms.ModelForm import ModelForm


@app.route("/", methods=["GET", "POST"])
def index():
    model_form = ModelForm()
    models = {field.name: field.data for field in model_form
                if field.type == "SelectField"}
    
    return render_template("index.html",
                           model_form=model_form,
                           models=models)
    

@app.route("/camera_stream", methods=["GET"])
def camera_stream():
    models= eval(request.args["models"]) # TODO: implement Model class; image processing
    camera = Camera()
    if not camera.is_available:
        return Response(camera.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return "/static/camera_not_found.jpg"