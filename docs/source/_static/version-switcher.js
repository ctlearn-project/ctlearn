(() => {
    const findManifest = async () => {
        const pathParts = window.location.pathname.split("/").filter(Boolean);
        for (let depth = 0; depth <= pathParts.length; depth += 1) {
            const candidate = new URL(
                `${"../".repeat(depth)}versions.json`,
                window.location.href,
            );
            try {
                const response = await fetch(candidate);
                if (!response.ok) {
                    continue;
                }
                const versions = await response.json();
                if (
                    Array.isArray(versions) &&
                    versions.some((version) => version.version === "main")
                ) {
                    return { candidate, versions };
                }
            } catch (error) {
                console.warn("Could not load documentation versions.", error);
            }
        }
        return null;
    };

    const addVersionSelector = async () => {
        const search = document.querySelector(".wy-side-nav-search");
        if (!search) {
            return;
        }

        const manifest = await findManifest();
        if (!manifest) {
            return;
        }

        const sitePath = new URL(".", manifest.candidate).pathname;
        const relativePath = window.location.pathname.slice(sitePath.length);
        const pathParts = relativePath.split("/").filter(Boolean);
        const currentVersion = manifest.versions.find(
            (version) => version.path === pathParts[0],
        );
        const pagePath = currentVersion ? pathParts.slice(1).join("/") : relativePath;
        const selectedVersion = currentVersion?.version ?? "main";

        const label = document.createElement("label");
        label.htmlFor = "documentation-version";
        label.textContent = "Documentation version";
        label.style.cssText = "display:block;margin:0.75em 0 0.25em;text-align:left";

        const select = document.createElement("select");
        select.id = "documentation-version";
        select.setAttribute("aria-label", "Documentation version");
        select.style.cssText = "box-sizing:border-box;width:100%;padding:0.35em";

        for (const version of manifest.versions) {
            const option = document.createElement("option");
            option.value = version.path;
            option.textContent = version.title;
            option.selected = version.version === selectedVersion;
            select.append(option);
        }

        select.addEventListener("change", () => {
            const destination = new URL(sitePath, window.location.origin);
            const targetPath = [select.value, pagePath].filter(Boolean).join("/");
            destination.pathname += targetPath;
            if (!pagePath || window.location.pathname.endsWith("/")) {
                destination.pathname += "/";
            }
            destination.search = window.location.search;
            destination.hash = window.location.hash;
            window.location.assign(destination);
        });

        search.append(label, select);
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", addVersionSelector);
    } else {
        addVersionSelector();
    }
})();
