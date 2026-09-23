from flask import Flask

app = Flask(__name__)

@app.route("/ping")
def ping():
    return {"message": "le serveur fonctionne"}

if __name__ == "__main__":
    app.run(port=5000)
