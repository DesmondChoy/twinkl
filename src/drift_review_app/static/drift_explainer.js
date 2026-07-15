(() => {
    const STEP_DELAY_MS = 3200;

    const initializeExplainer = (root) => {
        if (root.dataset.explainerReady === "true") return;

        const steps = Array.from(root.querySelectorAll(".explainer-step"));
        const partPanels = Array.from(
            root.querySelectorAll("[data-part-panel]")
        );
        const partTabs = Array.from(root.querySelectorAll("[data-part-target]"));
        const stepButtons = Array.from(root.querySelectorAll("[data-step-target]"));
        const previousButton = root.querySelector(
            '[data-explainer-action="previous"]'
        );
        const toggleButton = root.querySelector(
            '[data-explainer-action="toggle"]'
        );
        const nextButton = root.querySelector('[data-explainer-action="next"]');
        const progress = root.querySelector("[data-explainer-progress]");
        const reducedMotion = window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches;

        if (
            steps.length === 0 ||
            !previousButton ||
            !toggleButton ||
            !nextButton ||
            !progress
        ) {
            return;
        }

        root.dataset.explainerReady = "true";
        root.dataset.reducedMotion = reducedMotion ? "true" : "false";
        let currentIndex = Math.max(
            0,
            steps.findIndex((step) => !step.hidden)
        );
        let playing = false;
        let started = false;
        let timerId = null;

        const clearTimer = () => {
            if (timerId !== null) {
                window.clearTimeout(timerId);
                timerId = null;
            }
        };

        const render = () => {
            const activeStep = steps[currentIndex];
            const activePart = activeStep.dataset.part;

            partPanels.forEach((panel) => {
                panel.hidden = panel.dataset.partPanel !== activePart;
            });
            steps.forEach((step, index) => {
                step.hidden = index !== currentIndex;
            });
            partTabs.forEach((tab) => {
                const selected = tab.dataset.partTarget === activePart;
                tab.classList.toggle("is-active", selected);
                tab.setAttribute("aria-selected", selected ? "true" : "false");
                tab.tabIndex = selected ? 0 : -1;
            });
            stepButtons.forEach((button, index) => {
                const selected = index === currentIndex;
                button.classList.toggle("is-active", selected);
                if (selected) {
                    button.setAttribute("aria-current", "step");
                } else {
                    button.removeAttribute("aria-current");
                }
            });

            previousButton.disabled = currentIndex === 0;
            nextButton.disabled = currentIndex === steps.length - 1;
            toggleButton.textContent = playing
                ? "Pause sequence"
                : currentIndex === steps.length - 1
                  ? "Replay sequence"
                  : "Play sequence";
            toggleButton.setAttribute("aria-pressed", playing ? "true" : "false");

            progress.textContent = `Step ${currentIndex + 1} of ${steps.length}`;
        };

        const scheduleNext = () => {
            clearTimer();
            if (!root.isConnected) {
                playing = false;
                return;
            }
            if (!playing) return;
            if (currentIndex >= steps.length - 1) {
                playing = false;
                render();
                return;
            }
            timerId = window.setTimeout(() => {
                currentIndex += 1;
                render();
                scheduleNext();
            }, STEP_DELAY_MS);
        };

        const pause = () => {
            playing = false;
            clearTimer();
            render();
        };

        const goTo = (index) => {
            started = true;
            playing = false;
            clearTimer();
            currentIndex = Math.min(Math.max(index, 0), steps.length - 1);
            render();
        };

        previousButton.addEventListener("click", () => goTo(currentIndex - 1));
        nextButton.addEventListener("click", () => goTo(currentIndex + 1));
        toggleButton.addEventListener("click", () => {
            started = true;
            if (playing) {
                pause();
                return;
            }
            if (currentIndex === steps.length - 1) currentIndex = 0;
            playing = true;
            render();
            scheduleNext();
        });

        stepButtons.forEach((button, index) => {
            button.addEventListener("click", () => goTo(index));
        });

        partTabs.forEach((tab, tabIndex) => {
            tab.addEventListener("click", () => {
                const targetPart = tab.dataset.partTarget;
                const targetIndex = steps.findIndex(
                    (step) => step.dataset.part === targetPart
                );
                if (targetIndex >= 0) goTo(targetIndex);
            });
            tab.addEventListener("keydown", (event) => {
                if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
                event.preventDefault();
                const offset = event.key === "ArrowRight" ? 1 : -1;
                const targetIndex =
                    (tabIndex + offset + partTabs.length) % partTabs.length;
                partTabs[targetIndex].focus();
                partTabs[targetIndex].click();
            });
        });

        render();

        if (!reducedMotion && "IntersectionObserver" in window) {
            const observer = new IntersectionObserver(
                (entries) => {
                    if (started) {
                        observer.disconnect();
                        return;
                    }
                    if (!entries.some((entry) => entry.isIntersecting)) {
                        return;
                    }
                    started = true;
                    playing = true;
                    render();
                    scheduleNext();
                    observer.disconnect();
                },
                {threshold: 0.35}
            );
            observer.observe(root);
        }
    };

    const scan = () => {
        document
            .querySelectorAll("[data-drift-explainer]")
            .forEach(initializeExplainer);
    };

    const install = () => {
        scan();
        const observer = new MutationObserver(scan);
        observer.observe(document.body, {childList: true, subtree: true});
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", install, {once: true});
    } else {
        install();
    }
})();
