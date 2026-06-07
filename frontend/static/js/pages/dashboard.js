document.getElementById(
    "navbar"
).innerHTML =
    renderNavbar();

initializeTheme();

async function loadHealth() {

    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/health"
            );

        const data =
            await response.json();

        document.getElementById(
            "dashboard-metrics"
        ).innerHTML = `

            <div class="metric-card">

                <div class="metric-title">

                    🟢 Platform Status

                </div>

                <div class="metric-value status-green">

                    ${data.status}

                </div>

            </div>

            <div class="metric-card">

                <div class="metric-title">

                       Registered Models

                </div>

                <div class="metric-value">

                    ${data.registered_models}

                </div>

            </div>

            <div class="metric-card">

                <div class="metric-title">

                       Loaded Models

                </div>

                <div class="metric-value">

                    ${data.loaded_models}

                </div>

            </div>

            <div class="metric-card">

                <div class="metric-title">

                       Version

                </div>

                <div class="metric-value">

                    ${data.version}

                </div>

            </div>

        `;

    }

    catch (error) {

        console.error(
            error
        );

        document.getElementById(
            "dashboard-metrics"
        ).innerHTML =

            "Failed to load dashboard.";
    }
}

async function loadRecentModels() {

    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/models"
            );

        const models =
            await response.json();

        const container =
            document.getElementById(
                "recent-models"
            );

        if (
            models.length === 0
        ) {

            container.innerHTML = `

                <h2>

                    Recent Models

                </h2>

                <p>

                    No models registered.

                </p>

            `;

            return;
        }

        container.innerHTML = `

            <h2>

                Recent Models

            </h2>

            ${models
                .slice(-5)
                .reverse()
                .map(
                    model => `

                    <div class="recent-model">

                        <div class="recent-model-name">

                            ${model.name}

                        </div>

                        <div class="recent-model-meta">

                            ID: ${model.model_id}

                        </div>

                        <div class="recent-model-status">

                            ${model.status}

                        </div>

                    </div>

                `
                )
                .join("")
            }

        `;

    }

    catch (error) {

        console.error(
            error
        );

        document.getElementById(
            "recent-models"
        ).innerHTML = `

            <h2>

                Recent Models

            </h2>

            <p>

                Failed to load models.

            </p>

        `;
    }
}

loadHealth();

loadRecentModels();