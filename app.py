from flask import Flask, render_template, request
import pickle
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        study = int(request.form["study"])
        attendance = int(request.form["attendance"])
        marks = int(request.form["marks"])

        prediction = model.predict([[study, attendance, marks]])[0]

        # create graph
        values = [study, attendance, marks]
        labels = ["Study Hours", "Attendance", "Marks"]

        plt.bar(labels, values)
        plt.title("Student Input Analysis")
        plt.savefig("static/chart.png")
        plt.close()

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
