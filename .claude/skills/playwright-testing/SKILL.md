---
name: playwright-testing
description: Systematic Playwright MCP testing workflow with detailed checklists for visual testing and inspection
triggers:
  - test the app
  - playwright testing
  - visual testing
  - run playwright
---

# Playwright MCP Testing Guide

This skill provides a systematic checklist for visually testing and inspecting web apps using Playwright MCP.

---

## Workflow Instructions

**IMPORTANT**: Follow these instructions exactly:

1. **Use the Checklist System**: Work through the structured checklists systematically. Do NOT skip items or test ad-hoc.

2. **Track Progress**: Use your task/todo management tools to track which checklist items have been completed. Mark items as you complete them.

3. **Sequential Testing**: Work through the phases in order (Phase 1 → Phase 2 → Phase 3 → etc.). Within each phase, complete all checklist items before moving to the next phase.

4. **Document Everything**: For each checklist item:
   - Take a `browser_snapshot` or `browser_take_screenshot` as evidence
   - Note whether the item passed or failed
   - If failed, stop and follow the Bug Handling Workflow below

5. **Report Format**: When reporting results, use this format:
   ```
   ✅ [Item description] - PASSED
   ❌ [Item description] - FAILED: [brief description of issue]
   ```

6. **Per-Page Testing**: For multi-page apps, test at least 3-5 pages including the first page, a middle page, and the last page. Use the checklist template for EACH page tested.

---

## Bug Handling Workflow

**CRITICAL**: When any bug or visual error is detected during testing, follow this workflow exactly. Do NOT continue testing until the bug is fixed and verified.

1. **STOP testing immediately** - Do not continue to the next checklist item
2. **Document the bug** - Note the page number, steps to reproduce, and expected vs actual behavior
3. **Fix the bug** - Make the necessary code changes
4. **Restart the servers** (if applicable)
5. **Verify the fix with Playwright MCP**:
   - Navigate back to the same page/state where the bug occurred
   - Confirm the bug is resolved
   - Take a screenshot or snapshot as evidence
6. **Resume testing** from where you left off

This ensures bugs are caught and fixed immediately rather than accumulating a backlog of issues.

---

## Getting Started with Playwright MCP

### Navigation Commands
```
# Navigate to the app
mcp__playwright__browser_navigate → url: "http://localhost:5173"

# Take a snapshot (preferred over screenshot for accessibility)
mcp__playwright__browser_snapshot

# Take a screenshot for visual inspection
mcp__playwright__browser_take_screenshot

# Click an element (use ref from snapshot)
mcp__playwright__browser_click → element: "description", ref: "e123"

# Run custom JavaScript for complex interactions
mcp__playwright__browser_run_code → code: "async (page) => { ... }"
```

### Useful Patterns
```javascript
// Select text programmatically
await page.evaluate(() => {
  const element = document.querySelector('selector');
  const range = document.createRange();
  range.selectNodeContents(element);
  window.getSelection().removeAllRanges();
  window.getSelection().addRange(range);
  document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
});

// Check for element visibility
await page.locator('text=Some Text').isVisible();

// Wait for element
await page.waitForSelector('selector');

// IMPORTANT: Verify selection.toString() vs range.toString() consistency
// These can differ when content contains KaTeX or other complex DOM structures
await page.evaluate(() => {
  const element = document.querySelector('[class*="prose"] li');
  const range = document.createRange();
  range.selectNodeContents(element);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);

  // selection.toString() adds newlines between block elements
  // range.toString() matches actual text node content
  // TreeWalker counting should match range.toString()
  return {
    selectionLength: selection.toString().length,
    rangeLength: range.toString().length,
    textContentLength: element.textContent.length,
    // If these don't match, selection restoration may cause "bleeding"
    mismatch: selection.toString().length !== range.toString().length
  };
});
```

---

## Testing Workflow Template

### Phase 1: Initial Load Testing

#### Page Load
- [ ] Page loads without console errors
- [ ] Main heading/title is visible
- [ ] Core UI components render correctly
- [ ] No layout shifts or flickering

#### Navigation
- [ ] All navigation links work
- [ ] Active state shows correctly
- [ ] Mobile menu works (if applicable)

---

### Phase 2: Core Functionality Testing

**Repeat this checklist for each major feature/page:**

#### Feature: _______________

##### Layout & Components
- [ ] Layout renders correctly
- [ ] All interactive elements are accessible
- [ ] Responsive design works at different breakpoints
- [ ] Loading states display appropriately

##### User Interactions
- [ ] Click events work as expected
- [ ] Form inputs accept and validate data
- [ ] Keyboard navigation works
- [ ] Focus states are visible

##### Data & State
- [ ] Data loads and displays correctly
- [ ] State updates reflect in UI
- [ ] Error states handled gracefully
- [ ] Empty states display appropriately

---

### Phase 3: Cross-Feature Testing

#### State Persistence
- [ ] Refresh browser → state preserved (if applicable)
- [ ] Navigation between features preserves relevant state
- [ ] localStorage/sessionStorage works correctly

#### Error Handling
- [ ] Invalid inputs show error messages
- [ ] Network errors handled gracefully
- [ ] Console shows no unhandled errors

---

### Phase 4: Performance & Accessibility

#### Performance
- [ ] Initial page load < 3 seconds
- [ ] Interactions feel responsive
- [ ] No visible lag when scrolling
- [ ] No memory leaks during extended use

#### Accessibility
- [ ] All images have alt text
- [ ] Form fields have labels
- [ ] Color contrast is sufficient
- [ ] Screen reader announcements work

---

## Reporting Issues

When documenting bugs, include:
1. **Page/feature** where issue occurred
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Screenshot or snapshot** (use `mcp__playwright__browser_take_screenshot`)
6. **Console messages** (use `mcp__playwright__browser_console_messages`)

---

## Notes

- Playwright's mouse simulation may not perfectly replicate human interaction for text selection
- Use `browser_run_code` for complex JavaScript interactions
- Always take a `browser_snapshot` before clicking to get accurate element refs
- The `browser_snapshot` tool is preferred over screenshots for accessibility testing

---

## Quick Smoke Test Checklist

For rapid testing, verify these critical paths:

1. [ ] App loads without errors
2. [ ] Main functionality works
3. [ ] Can navigate between pages/features
4. [ ] Core user flows complete successfully
5. [ ] No console errors throughout
6. [ ] Responsive layout works on mobile viewport

---

## App-Specific Checklists

### Annotation Tool (`shiny run src/annotation_tool/app.py`)

**Navigation:** http://localhost:8000

#### Phase 1: Initial Load
- [ ] App loads showing persona list in left sidebar
- [ ] First entry content displays in center column
- [ ] Scoring grid (10 Schwartz values) visible in right column
- [ ] Annotator name input visible in header
- [ ] Progress bar shows 0% initially

#### Phase 2: Core Annotation Flow
- [ ] Enter annotator name and verify it persists
- [ ] Scoring buttons (−/+) increment/decrement values correctly
- [ ] Score values cycle through -1 → 0 → +1 → -1
- [ ] All 10 Schwartz values can be scored independently
- [ ] Persona bio toggle (Show Bio/Hide Bio) works

#### Phase 3: Post-Save Reveal Feature
- [ ] Click "Save & Next →" button
- [ ] **Inline comparison view replaces scoring grid** with:
  - [ ] "Score Comparison" title at top
  - [ ] Table showing all 10 Schwartz values
  - [ ] "You" column with human annotator scores
  - [ ] "Judge" column with LLM Judge scores
  - [ ] Match indicator column (✓/✗ only - no adjacent)
  - [ ] Color coding: green (exact match), red (disagree) - no yellow
  - [ ] Rationale rows visible below non-zero judge scores
  - [ ] Summary line (e.g., "7/10 exact matches")
- [ ] Inline view has "Continue →" button
- [ ] Clicking "Continue →" returns to scoring grid and advances to next entry
- [ ] Keyboard: Enter key triggers "Continue →" in comparison mode
- [ ] Progress bar updates after save

##### Rationale Display
- [ ] For non-zero judge scores, rationale row is always visible below the value row
- [ ] Rationale has distinct styling (italic, yellow-tinted background, left border)

##### Mode Switching
- [ ] Navigating to different entry resets to scoring mode
- [ ] Previous/Next persona buttons reset to scoring mode
- [ ] Clicking a different entry in sidebar resets to scoring mode

#### Phase 4: Missing Labels Handling
- [ ] For entries without Judge labels, inline view shows:
  - [ ] "No Judge labels available for this entry" message
  - [ ] Confirmation that annotation was saved
  - [ ] "Continue →" button still works

#### Phase 5: All-Neutral Warning
- [ ] Set all 10 scores to 0 (neutral)
- [ ] Click "Save & Next →"
- [ ] Warning modal appears asking to confirm all-neutral scores
- [ ] Can cancel and adjust scores
- [ ] Can confirm and proceed (then inline comparison view appears)

#### Phase 6: Re-edit Behavior
- [ ] Navigate to a previously annotated entry
- [ ] Existing scores load into scoring grid
- [ ] Modify one or more scores
- [ ] Save → inline comparison view replaces scoring grid (always shows on save)

#### Phase 7: Persistence
- [ ] Close browser/refresh page
- [ ] Verify progress persists (annotations saved to parquet)
- [ ] Verify annotated entries show checkmark (✓) in entry list
- [ ] Verify can resume from last position
