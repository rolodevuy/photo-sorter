"""Interfaz web (stub): revisión y corrección manual de detecciones.

Pendiente de implementar:
- Ver las fotos con los rostros detectados y sus nombres.
- Corregir asignaciones erróneas o etiquetar "desconocidos".
- Disparar el organizer desde la web.

Uso (cuando esté implementado):
    python -m app.flask_app
"""

from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "photo-sorter: interfaz web pendiente de implementar."


if __name__ == "__main__":
    app.run(debug=True)
