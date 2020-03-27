from flask import render_template
from flask import Response
from flask import request

from application import app
from application import MODELS
from application.Camera import Camera
from application.forms.ModelForm import ModelForm


@app.route("/", methods=["GET", "POST"])
def examples():
    return render_template("base.html")


@app.route("/live", methods=["GET", "POST"])
def live():
    model_form = ModelForm()
    models_selection = {field.name: field.data for field in model_form
                if field.type == "SelectField"}
    
    return render_template("live.html",
                           model_form=model_form,
                           models_selection=models_selection)
    

@app.route("/camera_stream", methods=["GET"])
def camera_stream():
    models_selection = eval(request.args["models_selection"])
    camera = Camera(detector=MODELS[models_selection["detector"]],
                    classifier=MODELS[models_selection["classifier"]])
    if camera.is_available:
        return Response(camera.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return "/static/camera_not_found.jpg"
