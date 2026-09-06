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
    const analyzeButton = document.getElementById("analyzeButton");

    error.textContent = "";

    if (!review) {
        error.textContent = "Please enter a review.";
        return;
    }

    loading.classList.remove("hidden");
    result.classList.add("hidden");
    analyzeButton.disabled = true;

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({
                review: review
            })
        });

        // Read the raw response first.
        const rawText = await response.text();

        // Check the HTTP status.
        if (!response.ok) {
            let serverMessage = "";

            try {
                const errorData = JSON.parse(rawText);

                if (errorData && errorData.error) {
                    serverMessage = errorData.error;
                }
            } catch (_) {
                serverMessage = rawText.substring(0, 300);
            }

            throw new Error(
                `Server error (${response.status})` +
                (serverMessage ? `: ${serverMessage}` : "")
            );
        }

        // Parse only after checking the raw response.
        let data;

        try {
            data = JSON.parse(rawText);
        } catch (_) {
            console.error("Non-JSON API response:", rawText);

            throw new Error(
                "The server did not return JSON. " +
                "Please check the Render deployment and API route."
            );
        }

        // Validate the API envelope.
        if (!data || data.success !== true) {
            throw new Error(
                data && data.error
                    ? data.error
                    : "Analysis failed"
            );
        }

        const resultData = data.result;

        if (!resultData) {
            throw new Error(
                "The server returned an incomplete analysis response."
            );
        }

        // ----------------------------------------------------
        // Prediction
        // ----------------------------------------------------

        document.getElementById("prediction").textContent =
            "Prediction: " +
            String(resultData.prediction || "")
                .toUpperCase();

        // ----------------------------------------------------
        // Fake probability
        // ----------------------------------------------------

        document.getElementById("fakeProbability").textContent =
            formatProbability(resultData.fake_probability);

        // ----------------------------------------------------
        // Risk level
        // ----------------------------------------------------

        document.getElementById("riskLevel").textContent =
            String(resultData.risk_level || "")
                .toUpperCase();

        // ----------------------------------------------------
        // Individual scores
        // ----------------------------------------------------

        document.getElementById("mlProbability").textContent =
            formatProbability(resultData.ml_probability);

        document.getElementById("semanticProbability").textContent =
            formatProbability(resultData.semantic_probability);

        document.getElementById("heuristicScore").textContent =
            formatProbability(resultData.heuristic_score);

        // ----------------------------------------------------
        // Reasons
        // ----------------------------------------------------

        const reasonsList = document.getElementById("reasons");
        reasonsList.innerHTML = "";

        const reasons = Array.isArray(resultData.reasons)
            ? resultData.reasons
            : [];

        if (reasons.length === 0) {
            const li = document.createElement("li");
            li.textContent = "No specific reasons were returned.";
            reasonsList.appendChild(li);
        } else {
            reasons.forEach((reason) => {
                const li = document.createElement("li");
                li.textContent = String(reason);
                reasonsList.appendChild(li);
            });
        }

        // Show result.
        result.classList.remove("hidden");

    } catch (errorObject) {
        console.error("Analysis error:", errorObject);

        error.textContent =
            errorObject instanceof Error
                ? errorObject.message
                : "An unexpected error occurred.";

    } finally {
        loading.classList.add("hidden");
        analyzeButton.disabled = false;
    }
}


/**
 * Convert a probability from 0-1 into a percentage string.
 */
function formatProbability(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "N/A";
    }

    return (numericValue * 100).toFixed(2) + "%";
}