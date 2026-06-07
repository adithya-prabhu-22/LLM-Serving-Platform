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

        <div
            style="
                margin-top:12px;
                display:flex;
                gap:10px;
            "
        >

            <button
                id="confirm-delete"
            >

                Delete

            </button>

            <button
                id="cancel-delete"
                class="delete-btn"
            >

                Cancel

            </button>

        </div>

    `;

  container.appendChild(toast);

  toast.querySelector("#confirm-delete").addEventListener(
    "click",

    () => {
      onConfirm();

      toast.remove();
    },
  );

  toast.querySelector("#cancel-delete").addEventListener(
    "click",

    () => {
      toast.remove();
    },
  );
}

async function loadModels() {
  try {
    const response = await fetch("http://127.0.0.1:8000/models");

    allModels = await response.json();

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

                            Architecture:
                            ${model.architecture}

                        </div>

                        <div>

                            <span
                                class="
                                    status-badge
                                    status-${model.status.toLowerCase()}
                                "
                            >

                                ${model.status}

                            </span>

                        </div>

                    </div>

                    <div class="model-actions">

                        <button
                            onclick="
                                buildModel(
                                    '${model.model_id}'
                                )
                            "
                        >

                            Build

                        </button>

                        <button
                            class="delete-btn"
                            onclick="
                                deleteModel(
                                    '${model.model_id}'
                                )
                            "
                        >

                            Delete

                        </button>

                    </div>

                </div>

            `,
    )
    .join("");
}

document.getElementById("search-models").addEventListener(
  "input",

  (event) => {
    const query = event.target.value.toLowerCase();

    const filtered = allModels.filter(
      (model) =>
        model.name.toLowerCase().includes(query) ||
        model.model_id.includes(query),
    );

    renderModels(filtered);
  },
);

async function buildModel(modelId) {
  try {
    const response = await fetch(
      `http://127.0.0.1:8000/models/build/${modelId}`,

      {
        method: "POST",
      },
    );

    const data = await response.json();

    showToast(data.message, "success");

    loadModels();
  } catch (error) {
    console.error(error);

    showToast("Failed to build model.", "error");
  }
}

async function deleteModel(modelId) {
  showConfirmToast(
    `Delete model '${modelId}'?`,

    async () => {
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/admin/models/${modelId}`,

          {
            method: "DELETE",
          },
        );

        const data = await response.json();

        showToast(data.message, "success");

        loadModels();
      } catch (error) {
        console.error(error);

        showToast("Failed to delete model.", "error");
      }
    },
  );
}

loadModels();
