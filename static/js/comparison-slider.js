/**
 * Before / After comparison slider.
 *
 * Progressively enhances a `.comparison-container` element:
 *   - Hides the side-by-side fallback grid.
 *   - Shows the overlay slider widget.
 *   - Supports range-input dragging AND pointer-drag on the divider.
 */
(function () {
    'use strict';

    const container = document.getElementById('comparisonSlider');
    const fallback  = document.getElementById('comparisonFallback');
    const slider    = document.getElementById('compSliderRange');
    const overlay   = document.getElementById('compOverlay');
    const divider   = document.getElementById('compDivider');
    const labelL    = document.getElementById('labelOriginal');
    const labelR    = document.getElementById('labelDetected');

    if (!container || !slider || !overlay) return; // graceful no-op

    /* ---- Show slider / hide fallback ---- */
    container.classList.remove('hidden');
    if (fallback) fallback.classList.add('hidden');

    /* ---- Core update ---- */
    function update(pct) {
        pct = Math.max(0, Math.min(100, pct));
        overlay.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
        divider.style.left = pct + '%';
        slider.value = pct;

        // Fade labels contextually
        if (labelL) labelL.style.opacity = pct < 15 ? 0 : 1;
        if (labelR) labelR.style.opacity = pct > 85 ? 0 : 1;
    }

    /* ---- Range input ---- */
    slider.addEventListener('input', () => update(parseFloat(slider.value)));

    /* ---- Pointer drag on divider / container ---- */
    let dragging = false;

    function pctFromEvent(e) {
        const rect = container.getBoundingClientRect();
        const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
        return (x / rect.width) * 100;
    }

    function onStart(e) {
        dragging = true;
        update(pctFromEvent(e));
        e.preventDefault();
    }
    function onMove(e) {
        if (!dragging) return;
        update(pctFromEvent(e));
        e.preventDefault();
    }
    function onEnd() { dragging = false; }

    divider.addEventListener('pointerdown', onStart);
    container.addEventListener('pointerdown', onStart);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onEnd);

    // Touch support
    container.addEventListener('touchstart', onStart, { passive: false });
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('touchend', onEnd);

    /* ---- Initial position ---- */
    update(50);
})();
