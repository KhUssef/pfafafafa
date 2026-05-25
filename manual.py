from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/")
def healthcheck():
	return jsonify({"status": "ok", "message": "Mock Flask app is running"})


@app.get("/scrape")
def scrape():
	# Mock response for a scrape operation.
	
		return jsonify({
				"success": True,
			"source": "mock",
			"results": [
				{"title": "Example Listing 1", "price": 120},
				{"title": "Example Listing 2", "price": 150},
				{"title": "Example Listing 3", "price": 100},
				{"title": "Example Listing 4", "price": 200},
                {"title": "Example Listing 5", "price": 80},
                {"title": "Example Listing 6", "price": 90},
                {"title": "Example Listing 7", "price": 110},
                {"title": "Example Listing 8", "price": 130},
                {"title": "Example Listing 9", "price": 140},
                {"title": "Example Listing 10", "price": 160},
			],})


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
