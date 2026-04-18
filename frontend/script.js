const API_BASE = "http://127.0.0.1:5000";

/* ------------------ DOM ELEMENTS ------------------ */
const chatMessages = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const loadingAnimation = document.getElementById("loadingAnimation");
const statusText = document.getElementById("statusText");
const doctorBtn = document.getElementById("doctorBtn");
const diseaseCardContainer = document.getElementById("diseaseCardContainer");

/* ------------------ EVENTS ------------------ */
sendBtn.addEventListener("click", sendMessage);
doctorBtn.addEventListener("click", startDoctorCall);

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});

/* ------------------ CHAT UI ------------------ */
function addMessage(sender, html) {
    const row = document.createElement("div");
    row.className = `chat-row ${sender}`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${sender}`;
    bubble.innerHTML = html;

    row.appendChild(bubble);
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/* ------------------ SEND MESSAGE ------------------ */
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    addMessage("user", text);
    userInput.value = "";

    loadingAnimation.style.display = "block";
    statusText.textContent = "Connecting to backend...";

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
        });

        const data = await res.json();
        loadingAnimation.style.display = "none";

        if (data.status !== "success") {
            addMessage("bot", "⚠️ Low confidence. Please add more symptoms.");
            return;
        }

        const main = data.top_predictions[0];

        // Chat output
        addMessage(
            "bot",
            `<b>${main.disease}</b> (${main.confidence}%)`
        );

        // ✅ Diagnosis Panel (NO WHY)
        diseaseCardContainer.innerHTML = `
            <div class="disease-card">
                <h3>${main.disease}</h3>

                <p><b>Confidence:</b> ${main.confidence}%</p>
                <p><b>Doctor Category:</b> ${main.doctor_category}</p>
                <p><b>Severity Score:</b> ${main.severity_score}</p>

                <hr>

                <p><b>Description:</b><br>${main.description || "Not available"}</p>
                <p><b>Prescription:</b><br>${main.prescription || "Consult a doctor"}</p>
                <p><b>Recommended Tests:</b><br>${main.tests || "Not required"}</p>
                <p><b>Precautions:</b><br>${main.precautions || "Take rest and monitor symptoms"}</p>
            </div>
        `;

        statusText.textContent = "Backend connected ✔";

    } catch (err) {
        console.error(err);
        loadingAnimation.style.display = "none";
        statusText.textContent = "Backend not connected ❌";
    }
}

/* ------------------ DOCTOR CALL ------------------ */
function startDoctorCall() {
    window.open(`${API_BASE}/doctor-call`, "_blank");
}
