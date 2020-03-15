from flask import render_template
from flask import Response

from application import app
from application.Camera import Camera
from application.forms.ModelForm import ModelForm


@app.route("/", methods=["GET"])
def index():
    model_form = ModelForm()
    
    if model_form.validate_on_submit():
        # TODO: implement validation logic
        pass

    return render_template("index.html", model_form=model_form)
    

@app.route("/camera_stream", methods=["GET"])
def camera_stream():
    camera = Camera()
    # TODO: implement Model class and model choice
    if not camera.is_available:
        return Response(camera.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return "/static/camera_not_found.jpg"