function renderNavbar() {
  return `

        <nav class="navbar">

            <div class="navbar-brand">

                LLM Serving Platform

            </div>

            <div class="navbar-links">

                <a href="index.html">
                    Dashboard
                </a>

                <a href="models.html">
                    Models
                </a>

                <a href="upload.html">
                    Upload
                </a>

                <a href="generate.html">
                    Generate
                </a>

                <button
    id="theme-toggle"
    class="theme-btn"
>
    Dark
</button>

            </div>

        </nav>

    `;
}
