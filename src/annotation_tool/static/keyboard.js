/**
 * Keyboard shortcut handler for the Schwartz Value Annotation Tool.
 *
 * Shortcuts:
 *   - Arrow Up/Down: Navigate between value rows
 *   - Arrow Left/Right: Decrement/Increment score
 *   - 1-9, 0: Jump to specific value (1=Self-Direction, 0=Universalism)
 *   - Enter: Save & Next
 *   - Backspace: Previous entry
 *   - ?: Toggle help modal
 *   - Escape: Clear focus / Close help
 */

document.addEventListener('DOMContentLoaded', function() {
    // Value order for keyboard navigation (matches SCHWARTZ_VALUE_ORDER)
    const valueOrder = [
        'self_direction', 'stimulation', 'hedonism', 'achievement',
        'power', 'security', 'conformity', 'tradition',
        'benevolence', 'universalism'
    ];

    let focusedRowIndex = -1;
    let helpVisible = false;

    /**
     * Update the focused row visual indicator.
     * @param {number} newIndex - Index of the row to focus (0-9)
     */
    function updateFocusedRow(newIndex) {
        // Remove previous focus
        document.querySelectorAll('.scoring-row.focused').forEach(el => {
            el.classList.remove('focused');
        });

        if (newIndex >= 0 && newIndex < valueOrder.length) {
            focusedRowIndex = newIndex;
            const value = valueOrder[focusedRowIndex];
            const row = document.getElementById('row_' + value);
            if (row) {
                row.classList.add('focused');
                row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    }

    /**
     * Cycle the score for the currently focused value.
     * @param {number} direction - -1 for decrement, 1 for increment
     */
    function cycleScore(direction) {
        if (focusedRowIndex < 0 || focusedRowIndex >= valueOrder.length) return;

        const value = valueOrder[focusedRowIndex];
        // direction: -1 = decrement, 1 = increment
        // Note: Button IDs are namespaced by Shiny module system with 'scoring-' prefix
        const btnId = 'scoring-' + (direction < 0 ? 'dec_' : 'inc_') + value;
        const btn = document.getElementById(btnId);
        if (btn) btn.click();
    }

    /**
     * Jump to a specific value by number key.
     * @param {number} num - Number key pressed (0-9)
     */
    function jumpToValue(num) {
        // 1-9 for first 9 values, 0 for 10th (Universalism)
        let index = num === 0 ? 9 : num - 1;
        if (index >= 0 && index < valueOrder.length) {
            updateFocusedRow(index);
        }
    }

    /**
     * Toggle the keyboard help modal visibility.
     */
    function toggleHelp() {
        const modal = document.getElementById('keyboard-help-modal');
        const backdrop = document.getElementById('keyboard-help-backdrop');
        if (modal && backdrop) {
            const newState = !helpVisible;
            modal.style.display = newState ? 'block' : 'none';
            backdrop.style.display = newState ? 'block' : 'none';
            helpVisible = newState;
        }
    }

    /**
     * Close the help modal if open.
     */
    function closeHelp() {
        const modal = document.getElementById('keyboard-help-modal');
        const backdrop = document.getElementById('keyboard-help-backdrop');
        if (modal && backdrop) {
            modal.style.display = 'none';
            backdrop.style.display = 'none';
            helpVisible = false;
        }
    }

    // Expose closeHelp globally for inline handlers and external calls
    window.closeHelp = closeHelp;

    /**
     * Clear row focus and visual indicators.
     */
    function clearFocus() {
        focusedRowIndex = -1;
        document.querySelectorAll('.scoring-row.focused').forEach(el => {
            el.classList.remove('focused');
        });
    }

    // Global keyboard handler
    document.addEventListener('keydown', function(e) {
        // Ignore if typing in an input field (except for Enter in annotator name)
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            // Allow Enter in annotator name field to blur and proceed
            if (e.key === 'Enter' && e.target.id === 'annotator_name') {
                e.target.blur();
                return;
            }
            return;
        }

        switch(e.key) {
            case 'ArrowUp':
                e.preventDefault();
                updateFocusedRow(focusedRowIndex <= 0 ? valueOrder.length - 1 : focusedRowIndex - 1);
                break;

            case 'ArrowDown':
                e.preventDefault();
                updateFocusedRow(focusedRowIndex >= valueOrder.length - 1 ? 0 : focusedRowIndex + 1);
                break;

            case 'ArrowLeft':
                e.preventDefault();
                cycleScore(-1);
                break;

            case 'ArrowRight':
                e.preventDefault();
                cycleScore(1);
                break;

            case 'Enter':
                e.preventDefault();
                // Try save button first (scoring mode), then continue button (comparison mode)
                // Note: save_btn is namespaced (scoring-save_btn), but continue_btn is not
                const saveBtn = document.getElementById('scoring-save_btn');
                const continueBtn = document.getElementById('continue_btn');
                if (saveBtn) saveBtn.click();
                else if (continueBtn) continueBtn.click();
                break;

            case 'Backspace':
                e.preventDefault();
                // Note: Button ID is namespaced by Shiny module system (header-prev_btn)
                const prevBtn = document.getElementById('header-prev_btn');
                if (prevBtn) prevBtn.click();
                break;

            case '?':
                e.preventDefault();
                toggleHelp();
                break;

            case 'Escape':
                if (helpVisible) {
                    closeHelp();
                }
                clearFocus();
                break;

            default:
                // Number keys 0-9 for jumping to values
                if (e.key >= '0' && e.key <= '9') {
                    e.preventDefault();
                    jumpToValue(parseInt(e.key));
                }
        }
    });

    // Initialize focus on first row
    if (focusedRowIndex === -1) {
        focusedRowIndex = 0;
    }

    // Help button click handler - using event delegation since Shiny renders after DOMContentLoaded
    document.addEventListener('click', function(e) {
        if (e.target.closest('#header-help_btn')) {
            e.preventDefault();
            toggleHelp();
        }
    });

    // Entry click handler - using event delegation for sidebar and entry cards
    // Uses nonce to ensure Shiny reactive event fires even if same entry is clicked
    document.addEventListener('click', function(e) {
        // Check for sidebar entry items first
        const sidebarItem = e.target.closest('.sidebar-entry-item');
        if (sidebarItem && !sidebarItem.classList.contains('locked')) {
            const entryIndex = sidebarItem.getAttribute('data-entry-index');
            if (entryIndex !== null) {
                // Include nonce to force reactive event even if index unchanged
                Shiny.setInputValue('selected_entry_index', {
                    index: parseInt(entryIndex),
                    nonce: Date.now()
                }, {priority: 'event'});
            }
            return;
        }

        // Fall back to legacy entry card handling
        const entryCard = e.target.closest('.entry-card');
        if (entryCard && !entryCard.classList.contains('locked')) {
            const entryIndex = entryCard.getAttribute('data-entry-index');
            if (entryIndex !== null) {
                // Include nonce to force reactive event even if index unchanged
                Shiny.setInputValue('selected_entry_index', {
                    index: parseInt(entryIndex),
                    nonce: Date.now()
                }, {priority: 'event'});
            }
        }
    });

    // ============================================
    // DARK MODE TOGGLE
    // ============================================

    /**
     * Initialize theme from localStorage on page load.
     */
    function initTheme() {
        const saved = localStorage.getItem('theme');
        if (saved === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
            updateThemeIcon(true);
        }
    }

    /**
     * Toggle between light and dark themes.
     */
    function toggleTheme() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        if (isDark) {
            document.documentElement.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
        } else {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        }
        updateThemeIcon(!isDark);
    }

    /**
     * Update the theme toggle icon based on current theme.
     * @param {boolean} isDark - Whether dark mode is active
     */
    function updateThemeIcon(isDark) {
        const icon = document.querySelector('.theme-icon');
        if (icon) {
            icon.textContent = isDark ? '☀️' : '🌙';
        }
    }

    // Initialize theme on load
    initTheme();

    // Listen for theme toggle button clicks
    document.addEventListener('click', function(e) {
        if (e.target.closest('.theme-toggle')) {
            toggleTheme();
        }
    });
});
