"""Régénère app/index.html : python -m app.build (depuis la racine du repo).

1. src/app_export.build() exporte app/app_data.json à partir des modèles et de reports/
   (exécuter avant : notebooks 02_interpretability, 06_economics, 07_tradeoffs) ;
2. vérifie que les scores recalculés comme dans le navigateur sont ceux de Python ;
3. injecte les données et app/score.js dans app/index.template.html.
"""
from pathlib import Path

from src.app_export import build, load_context, verify_parity

APP = Path(__file__).resolve().parent

if __name__ == "__main__":
    data, test, pipe, lf = build()
    ctx = load_context()
    ex, el = verify_parity(data, test, ctx["xgb"], pipe, lf, ctx["features"])
    assert ex < 1e-5 and el < 1e-5, f"parité rompue : xgb {ex}, logit {el}"
    page = (APP / "index.template.html").read_text()
    page = page.replace("__DATA__", (APP / "app_data.json").read_text().replace("</", "<\\/"))
    page = page.replace("__SCORE__", (APP / "score.js").read_text())
    (APP / "index.html").write_text(page)
    print(f"app/index.html régénéré ({len(page) / 1e6:.1f} Mo), parité xgb {ex:.1e} / logit {el:.1e}")
