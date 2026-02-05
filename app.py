from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ---------------- BASIC SETUP ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------- DATABASE MODEL ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("best_8class_model.h5")

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Check existing user
        if User.query.filter_by(username=username).first():
            flash("Username already exists")
            return redirect(url_for("signup"))

        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()

        flash("Signup successful. Please login.")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and user.password == password:
            session["user_id"] = user.id
            return redirect(url_for("prediction"))
        else:
            flash("Invalid username or password")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully")
    return redirect(url_for("home"))

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user_id" not in session:
        flash("Please login first")
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])
    result = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file)
            img = preprocess_image(img)
            prediction = model.predict(img)
            idx = np.argmax(prediction)

            classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
            result = classes[idx]

            return jsonify({"prediction": result})

    return render_template("prediction.html", username=user.username, prediction=result)

# ---------------- MAIN ----------------
# ---------------- MAIN ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(host="0.0.0.0", port=10000)



