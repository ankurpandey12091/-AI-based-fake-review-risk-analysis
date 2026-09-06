function useSample(type) {
    const reviewBox = document.getElementById("review");

    if (type === "fake") {
        reviewBox.value =
            "AMAZING!!! BEST PRODUCT EVER!!! BUY NOW!!! FIVE STARS!!!";
    } else {
        reviewBox.value =
            "The battery lasted three days with normal use. " +
            "Setup was easy and the instructions were clear.";
    }
}


async function analyzeReview() {
    const review = document.getElementById("review").value.trim();

    const error = document.getElementById("error");
    const result = document.getElementById("result");
    const loading = document.getElementById("loading");

    error.textContent = "";

    if (!review) {
        error.textContent = "Please enter a review.";
        return;
    }

    loading.classList.remove("hidden");
    result.classList.add("hidden");

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                review: review
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Analysis failed");
        }

        const resultData = data.result;

        document.getElementById("prediction").textContent =
            "Prediction: " + resultData.prediction.toUpperCase();

        document.getElementById("fakeProbability").textContent =
            (resultData.fake_probability * 100).toFixed(2) + "%";

        document.getElementById("riskLevel").textContent =
            resultData.risk_level.toUpperCase();

        document.getElementById("mlProbability").textContent =
            (resultData.ml_probability * 100).toFixed(2) + "%";

        document.getElementById("semanticProbability").textContent =
            (resultData.semantic_probability * 100).toFixed(2) + "%";

        document.getElementById("heuristicScore").textContent =
            (resultData.heuristic_score * 100).toFixed(2) + "%";

        const reasonsList = document.getElementById("reasons");
        reasonsList.innerHTML = "";

        resultData.reasons.forEach(reason => {
            const li = document.createElement("li");
            li.textContent = reason;
            reasonsList.appendChild(li);
        });

        result.classList.remove("hidden");

    } catch (errorObject) {
        error.textContent = errorObject.message;
    } finally {
        loading.classList.add("hidden");
    }
}