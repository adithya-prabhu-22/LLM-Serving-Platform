document.getElementById("navbar").innerHTML = renderNavbar();

initializeTheme();

const modelSelect = document.getElementById("model-id");

const promptInput = document.getElementById("prompt");

const maxTokensInput = document.getElementById("max-tokens");

const generateButton = document.getElementById("generate-btn");

const clearButton = document.getElementById("clear-prompt");

const copyButton = document.getElementById("copy-output");

const outputBox = document.getElementById("output-box");

const outputModel = document.getElementById("output-model");

const outputTokens = document.getElementById("output-tokens");

const modelStatus = document.getElementById("model-status");

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

async function loadModels() {
  try {
    const response = await fetch("http://127.0.0.1:8000/models");

    const models = await response.json();

    const readyModels = models.filter((model) => model.status === "READY");

    if (readyModels.length === 0) {
      modelSelect.innerHTML = `

                <option value="">

                    No READY Models

                </option>

            `;

      modelStatus.textContent = "Status: No READY models";

      return;
    }

    modelSelect.innerHTML = readyModels
      .map(
        (model) => `

                        <option
                            value="${model.model_id}"
                            data-status="${model.status}"
                        >

                            ${model.name}

                        </option>

                    `,
      )
      .join("");

    modelStatus.textContent = "Status: READY";
  } catch (error) {
    console.error(error);

    showToast("Failed to load models", "error");
  }
}

clearButton.addEventListener(
  "click",

  () => {
    promptInput.value = "";

    promptInput.focus();
  },
);

copyButton.addEventListener(
  "click",

  async () => {
    try {
      await navigator.clipboard.writeText(outputBox.textContent);

      showToast("Output copied");
    } catch {
      showToast("Failed to copy", "error");
    }
  },
);

generateButton.addEventListener(
  "click",

  async () => {
    const prompt = promptInput.value.trim();

    if (!prompt) {
      showToast("Prompt is required", "error");

      return;
    }

    if (!modelSelect.value) {
      showToast("Select a model", "error");

      return;
    }

    outputBox.textContent = "Generating...";

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/generate",

        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            model_id: modelSelect.value,

            prompt: prompt,

            max_new_tokens: parseInt(maxTokensInput.value),
          }),
        },
      );

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Generation failed");
      }

      outputBox.textContent = result.response;

      outputModel.textContent =
        modelSelect.options[modelSelect.selectedIndex].text;

      outputTokens.textContent = maxTokensInput.value;

      showToast("Generation completed");
    } catch (error) {
      console.error(error);

      outputBox.textContent = "Generation failed.";

      showToast(error.message, "error");
    }
  },
);

loadModels();
