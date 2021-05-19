from flask import Flask, render_template, request, redirect
import speech_recognition as sr
import moviepy.editor as mp

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            clip = mp.VideoFileClip(r"video_recording.mp4")

            clip.audio.write_audiofile(r"converted.wav")
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile("converted.wav")
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)

    return render_template("index.html", transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
