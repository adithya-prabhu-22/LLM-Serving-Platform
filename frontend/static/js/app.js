function initializeTheme() {

    const savedTheme =
        localStorage.getItem(
            "theme"
        );

    if (
        savedTheme ===
        "dark"
    ) {

        document.body.classList.add(
            "dark-theme"
        );
    }

    updateThemeButton();

    const themeButton =
        document.getElementById(
            "theme-toggle"
        );

    if (!themeButton) {

        return;
    }

    themeButton.addEventListener(

        "click",

        () => {

            document.body.classList.toggle(
                "dark-theme"
            );

            const isDark =
                document.body.classList.contains(
                    "dark-theme"
                );

            localStorage.setItem(

                "theme",

                isDark
                    ? "dark"
                    : "light"
            );

            updateThemeButton();
        }
    );
}

function updateThemeButton() {

    const themeButton =
        document.getElementById(
            "theme-toggle"
        );

    if (!themeButton) {

        return;
    }

    const isDark =
        document.body.classList.contains(
            "dark-theme"
        );

    themeButton.textContent =
        isDark
            ? "Light"
            : "Dark";
}