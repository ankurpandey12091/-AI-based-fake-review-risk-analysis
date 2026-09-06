from flask import Flask, jsonify, render_template, request

from analyzer.hybrid_detector import analyze_review


app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "AI Fake Review Monitor"
    })


@app.post("/api/analyze")
def analyze():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request must contain JSON"
        }), 400

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Invalid JSON body"
        }), 400

    review = data.get("review")

    if not isinstance(review, str):
        return jsonify({
            "success": False,
            "error": "review must be a string"
        }), 400

    review = review.strip()

    if not review:
        return jsonify({
            "success": False,
            "error": "review cannot be empty"
        }), 400

    if len(review) > 5000:
        return jsonify({
            "success": False,
            "error": "review is too long"
        }), 400

    try:
        result = analyze_review(review)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as exc:
        app.logger.exception("Review analysis failed")

        return jsonify({
            "success": False,
            "error": "Unable to analyze review"
        }), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


if __name__ == "__main__":
    app.run(debug=True)