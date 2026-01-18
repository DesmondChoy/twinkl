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
        const btnId = direction < 0 ? 'dec_' + value : 'inc_' + value;
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
        if (modal && backdrop && helpVisible) {
            modal.style.display = 'none';
            backdrop.style.display = 'none';
            helpVisible = false;
        }
    }

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
                const saveBtn = document.getElementById('save_btn');
                if (saveBtn) saveBtn.click();
                break;

            case 'Backspace':
                e.preventDefault();
                const prevBtn = document.getElementById('prev_btn');
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

    // Help button click handler
    const helpBtn = document.getElementById('help_btn');
    if (helpBtn) {
        helpBtn.addEventListener('click', function(e) {
            e.preventDefault();
            toggleHelp();
        });
    }

    // Entry click handler - using event delegation for sidebar and entry cards
    document.addEventListener('click', function(e) {
        // Check for sidebar entry items first
        const sidebarItem = e.target.closest('.sidebar-entry-item');
        if (sidebarItem && !sidebarItem.classList.contains('locked')) {
            const entryIndex = sidebarItem.getAttribute('data-entry-index');
            if (entryIndex !== null) {
                Shiny.setInputValue('selected_entry_index', parseInt(entryIndex), {priority: 'event'});
            }
            return;
        }

        // Fall back to legacy entry card handling
        const entryCard = e.target.closest('.entry-card');
        if (entryCard && !entryCard.classList.contains('locked')) {
            const entryIndex = entryCard.getAttribute('data-entry-index');
            if (entryIndex !== null) {
                Shiny.setInputValue('selected_entry_index', parseInt(entryIndex), {priority: 'event'});
            }
        }
    });
});
