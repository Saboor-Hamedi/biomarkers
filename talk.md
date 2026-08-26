# TASK: Fix Layout Constraints for "No-Scroll" Dashboard View

## Objective
Refactor the current "Forensic Overview" page to fit entirely within the viewport height (`100vh`). 
**CRITICAL:** The main page must NEVER scroll. All content must be visible immediately without scrolling down.

## Layout Architecture Requirements

### 1. Global Container Constraints
- Apply `h-screen` and `overflow-hidden` to the main application wrapper.
- The content area (right of the sidebar) must use `flex flex-col` with `h-full`.
- Ensure the header, metrics row, calibration section, and parameters section all fit within this fixed height.

### 2. Section Sizing & Distribution
- **Header:** Fixed height (e.g., `h-16`), shrink-0.
- **Metrics Row (Top 3 Cards):** Fixed height or compact padding. Do not let these expand unnecessarily.
- **Calibration Section (Middle):** This is the flexible element. 
    - It must use `flex-1` or `min-h-0` to occupy remaining space but NOT push content off-screen.
    - If the file list is long, the **internal list area** must scroll (`overflow-y-auto`), NOT the whole page.
    - Reduce vertical padding/margins inside this card to save space.
- **Forensic Parameters (Bottom):** Fixed height (e.g., `h-auto` or specific pixel height), shrink-0. It must stick to the bottom of the view.

### 3. Visual Adjustments for Compactness
- Reduce gap spacing between major sections (e.g., change `gap-8` to `gap-4`).
- Tighten padding inside cards (e.g., change `p-8` to `p-5` or `p-6`).
- Ensure the "Drop Model Files Here" zone is not excessively tall.

## Technical Implementation Details
- Use Tailwind CSS utility classes for all sizing.
- **Forbidden:** Do not use `min-h-screen` (which allows growth). Use `h-screen` (fixed).
- **Forbidden:** Do not allow the body or html tags to have `overflow: auto`.
- **Required:** The "Calibration" card's file list grid must have a defined `max-height` (e.g., `max-h-[300px]`) and `overflow-y-auto` so that if there are many files, only that small list scrolls, keeping the rest of the UI static.

## Deliverable
A refactored layout component where:
1. I can see the Header, 3 Metrics, Calibration Dropzone, File List, AND Forensic Parameters simultaneously.
2. Scrolling the mouse wheel does NOT move the page up/down.
3. Only the internal file list area scrolls if content overflows.