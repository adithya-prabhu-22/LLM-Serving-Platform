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

const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_TOP_K = 40;

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
    const response = await fetch("/list-s3-models");
    const data = await response.json();
    const models = data.models || [];

    if (models.length === 0) {
      modelSelect.innerHTML = `
        <option value="">
          No Models Available
        </option>
      `;
      modelStatus.textContent = "Status: No models in S3";
      return;
    }

    modelSelect.innerHTML = models
      .map(
        (name) => `
            <option value="${name}" data-status="AVAILABLE">
              ${name}
            </option>
          `,
      )
      .join("");

    modelStatus.textContent = "Status: AVAILABLE";
  } catch (error) {
    console.error(error);
    showToast("Failed to load models", "error");
  }
}

async function ensureModelLoaded(modelName) {
  try {
    const response = await fetch(`/load-s3-model/${modelName}`, {
      method: "POST",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Failed to load model");
    }
    return true;
  } catch (error) {
    console.error(error);
    showToast(error.message || "Failed to load model", "error");
    return false;
  }
}

clearButton.addEventListener("click", () => {
  promptInput.value = "";
  promptInput.focus();
});

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(outputBox.textContent);
    showToast("Output copied");
  } catch {
    showToast("Failed to copy", "error");
  }
});

generateButton.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();

  if (!prompt) {
    showToast("Prompt is required", "error");
    return;
  }

  if (!modelSelect.value) {
    showToast("Select a model", "error");
    return;
  }

  generateButton.disabled = true;
  generateButton.textContent = "Generating...";
  outputBox.textContent = "";
  outputModel.textContent = modelSelect.options[modelSelect.selectedIndex].text;
  outputTokens.textContent = "0";

  try {
    modelStatus.textContent = "Status: LOADING";
    const loaded = await ensureModelLoaded(modelSelect.value);
    if (!loaded) {
      throw new Error("Model could not be loaded");
    }
    modelStatus.textContent = "Status: READY";

    const maxTokens = Math.min(parseInt(maxTokensInput.value) || 200, 200);

    const response = await fetch("/generate/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_id: modelSelect.value,
        prompt,
        max_new_tokens: maxTokens,
        temperature: DEFAULT_TEMPERATURE,
        top_k: DEFAULT_TOP_K,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let tokensGenerated = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      const chunk = decoder.decode(value, { stream: true });
      outputBox.textContent += chunk;
      tokensGenerated += 1;
      outputTokens.textContent = tokensGenerated.toString();
    }

    showToast("Generation completed");
  } catch (error) {
    console.error(error);
    outputBox.textContent = "Generation failed.";
    showToast(error.message, "error");
  } finally {
    generateButton.disabled = false;
    generateButton.textContent = "Generate";
  }
});

loadModels();