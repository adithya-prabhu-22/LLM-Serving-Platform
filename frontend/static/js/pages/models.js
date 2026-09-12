document.getElementById("navbar").innerHTML = renderNavbar();

initializeTheme();

let allModels = [];

function showToast(message, type = "success") {
  const container = document.getElementById("notification-container");
  const toast = document.createElement("div");
  toast.className = `notification-toast notification-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3000);
}

function showConfirmToast(message, onConfirm) {
  const container = document.getElementById("notification-container");
  const toast = document.createElement("div");
  toast.className = "notification-toast";
  toast.innerHTML = `
        <div>
            ${message}
        </div>
        <div style="margin-top:12px;display:flex;gap:10px;">
            <button id="confirm-delete">Delete</button>
            <button id="cancel-delete" class="delete-btn">Cancel</button>
        </div>
    `;
  container.appendChild(toast);
  toast.querySelector("#confirm-delete").addEventListener("click", () => {
    onConfirm();
    toast.remove();
  });
  toast.querySelector("#cancel-delete").addEventListener("click", () => {
    toast.remove();
  });
}

async function loadModels() {
  try {
    const response = await fetch("/list-s3-models");
    const data = await response.json();
    allModels = (data.models || []).map((name) => ({
      model_id: name,
      name: name,
      architecture: "GPT",
      status: "AVAILABLE",
    }));
    renderModels(allModels);
  } catch (error) {
    document.getElementById("models-container").innerHTML =
      "Failed to load models.";
    console.error(error);
  }
}

function renderModels(models) {
  const container = document.getElementById("models-container");
  if (models.length === 0) {
    container.innerHTML = `
            <div class="card">
                No models registered.
            </div>
        `;
    return;
  }
  container.innerHTML = models
    .map(
      (model) => `
                <div class="model-card">
                    <div>
                        <div class="model-name">
                            ${model.name}
                        </div>
                        <div class="model-meta">
                            ID: ${model.model_id}
                        </div>
                        <div class="model-meta">
                            Architecture: ${model.architecture}
                        </div>
                        <div>
                            <span class="status-badge status-${model.status.toLowerCase()}">
                                ${model.status}
                            </span>
                        </div>
                    </div>
                    <div class="model-actions">
                        <button onclick="loadModel('${model.model_id}')">
                            Load
                        </button>
                    </div>
                </div>
            `,
    )
    .join("");
}

document.getElementById("search-models").addEventListener("input", (event) => {
  const query = event.target.value.toLowerCase();
  const filtered = allModels.filter(
    (model) =>
      model.name.toLowerCase().includes(query) ||
      model.model_id.toLowerCase().includes(query),
  );
  renderModels(filtered);
});

async function loadModel(modelId) {
  try {
    showToast(`Loading model '${modelId}'...`, "success");
    const response = await fetch(`/load-s3-model/${modelId}`, {
      method: "POST",
    });
    const data = await response.json();
    if (response.ok) {
      showToast(data.message, "success");
    } else {
      showToast(data.detail || "Failed to load model.", "error");
    }
  } catch (error) {
    console.error(error);
    showToast("Failed to load model.", "error");
  }
}

loadModels();
