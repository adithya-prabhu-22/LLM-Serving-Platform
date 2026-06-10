document.getElementById("navbar").innerHTML = renderNavbar();

initializeTheme();

const uploadForm = document.getElementById("upload-form");

uploadForm.addEventListener(
  "submit",

  async function (event) {
    event.preventDefault();

    const resultBox = document.getElementById("upload-result");

    const formData = new FormData();

    formData.append(
      "model_id",

      document.getElementById("model-id").value,
    );

    formData.append(
      "name",

      document.getElementById("name").value,
    );

    formData.append(
      "architecture",

      document.getElementById("architecture").value,
    );

    formData.append(
      "config_file",

      document.getElementById("config-file").files[0],
    );

    formData.append(
      "weights_file",

      document.getElementById("weights-file").files[0],
    );

    resultBox.innerHTML = `

        <div
            class="notification"
        >

            Uploading model...

        </div>

    `;

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/admin/models/upload",

        {
          method: "POST",

          body: formData,
        },
      );

      const result = await response.json();

      if (response.ok) {
        resultBox.innerHTML = `

            <div
                class="
                    notification
                    notification-success
                "
            >

                ${result.message}

            </div>

        `;

        uploadForm.reset();

        document.getElementById("config-file-name").textContent =
          "No file selected";

        document.getElementById("weights-file-name").textContent =
          "No file selected";
      } else {
        resultBox.innerHTML = `

            <div
                class="
                    notification
                    notification-error
                "
            >

                ${result.detail || result.message || "Upload Failed"}

            </div>

        `;
      }
    } catch (error) {
      console.error(error);

      resultBox.innerHTML = `

            <div
                class="
                    notification
                    notification-error
                "
            >

                Failed to connect to backend.

            </div>

        `;
    }
  },
);

document.getElementById("config-file").addEventListener(
  "change",

  function () {
    document.getElementById("config-file-name").textContent =
      this.files.length ? this.files[0].name : "No file selected";
  },
);

document.getElementById("weights-file").addEventListener(
  "change",

  function () {
    document.getElementById("weights-file-name").textContent =
      this.files.length ? this.files[0].name : "No file selected";
  },
);
