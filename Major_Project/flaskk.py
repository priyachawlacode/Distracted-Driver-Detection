from flask import Flask,render_template,request,redirect, url_for
from werkzeug.utils import secure_filename
from predict import predict

app = Flask(__name__)

tags = ["safe driving","texting_right","talking on the phone_right","texting_left","talking on the phone_left","operating the radio","drinking","reaching behind","hair and makeup","talking to passenger"]


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detect', methods = ['GET', 'POST'])
def detect():
	modal=safe=False
	if request.method == 'POST':
		f = request.files['file']
		f.filename="user_img.jpg"
		f.save("Test/"+f.filename)
		answer=predict()
		print(tags[answer])
		if answer==0:
			safe=True
		modal=True
		return render_template("index.html",ans=tags[answer].title(),modal=modal,safe=safe)
	return render_template("index.html")


if __name__ == '__main__':
    app.run(host="localhost", port=7000, debug=True)