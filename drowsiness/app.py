
from flask import Flask, render_template, request, jsonify
import os
import mysql.connector
from detector import detect_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# DATABASE CONNECTION
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Geetha3230",
    database="drowsiness"
)

cursor = db.cursor()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload_image", methods=["POST"])
def upload_image():

    if "file" not in request.files:
        return jsonify({"result": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"result": "No file selected"})

    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(path)

    # DETECTION
    result = detect_image(path)

    # SAVE TO DB
    sql = """
    INSERT INTO detection_results (file_name, prediction)
    VALUES (%s, %s)
    """
    cursor.execute(sql, (filename, result))
    db.commit()

    # DELETE FILE AFTER PROCESSING
    os.remove(path)

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=False)