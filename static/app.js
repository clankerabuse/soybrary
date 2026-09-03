/* ─── DOM refs ────────────────────────────────────────────────────────────────── */
const grid           = document.getElementById('grid');
const searchInput    = document.getElementById('search');
const searchClear    = document.getElementById('search-clear');
const autocomplete   = document.getElementById('autocomplete');
const modal          = document.getElementById('modal');
const modalImg       = document.getElementById('modal-img');
const modalMeta      = document.getElementById('modal-meta');
const modalIdBadge   = document.getElementById('modal-id-badge');
const modalOpenLink  = document.getElementById('modal-open-link');
const statsEl        = document.getElementById('stats');
const paginationEl   = document.getElementById('pagination');
const pageJumpEl     = document.getElementById('page-jump');
const pageJumpForm   = document.getElementById('page-jump-form');
const pageJumpInput  = document.getElementById('page-jump-input');
const pageJumpHint   = document.getElementById('page-jump-hint');
const scrapeBtn      = document.getElementById('scrape-btn');
const scrapeBtnLabel = scrapeBtn.querySelector('.btn-label');
const scrapeBtnIconPlay = scrapeBtn.querySelector('.btn-icon-play');
const scrapeBtnIconStop = scrapeBtn.querySelector('.btn-icon-stop');
const consoleToggle  = document.getElementById('console-toggle');
const scrapeConsole  = document.getElementById('scrape-console');
const consoleOutput  = document.getElementById('console-output');
const consoleClear   = document.getElementById('console-clear');
const consoleClose   = document.getElementById('console-close');
const consoleDot     = scrapeConsole.querySelector('.console-dot');
const loadingIndicator = document.getElementById('loading-indicator');
const emptyState     = document.getElementById('empty-state');
const toastContainer = document.getElementById('toast-container');

/* ─── State ───────────────────────────────────────────────────────────────────── */
let currentPage  = 1;
let currentQuery = '';
let totalPages   = 1;
let totalPosts   = 0;
let isLoading    = false;
let maxId        = 0;
let isScraping   = false;
let selectedIndex = -1;
let autocompleteAbortController = null;
let autocompleteRequestId = 0;

const LIMIT = 50;

/* Animated previews are only fetched once they are near the viewport — a grid
   page holds 50 posts and the originals are far heavier than the thumbnails. */
const previewObserver = new IntersectionObserver(
    (entries) => {
        for (const entry of entries) {
            const media = entry.target;
            if (entry.isIntersecting) {
                activatePreview(media);
            } else if (media.tagName === 'VIDEO') {
                media.pause();
            }
        }
    },
    { threshold: 0.15, rootMargin: '200px' }
);

function activatePreview(media) {
    const full = media.dataset.previewSrc;
    if (full) {
        delete media.dataset.previewSrc;
        media.src = full;
    }
    if (media.tagName === 'VIDEO') media.play().catch(() => {});
}

/* ─── Toast ───────────────────────────────────────────────────────────────────── */
function showToast(message, type = 'info', duration = 4000) {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `<span class="toast-dot"></span><span>${escapeHtml(message)}</span>`;
    toastContainer.appendChild(t);

    setTimeout(() => {
        t.classList.add('leaving');
        t.addEventListener('animationend', () => t.remove(), { once: true });
    }, duration);
}

/* ─── Skeleton ────────────────────────────────────────────────────────────────── */
function showSkeletons(count = 20) {
    for (let i = 0; i < count; i++) {
        const div = document.createElement('div');
        div.className = 'skeleton-item';
        // Vary heights for visual rhythm
        const h = 100 + Math.floor(Math.random() * 140);
        div.innerHTML = `<div class="skeleton-inner" style="height:${h}px"></div>`;
        grid.appendChild(div);
    }
}

function clearSkeletons() {
    grid.querySelectorAll('.skeleton-item').forEach(el => el.remove());
}

/* ─── Helpers ────────────────────────────────────────────────────────────────── */
function absUrl(path) {
    return new URL(path, window.location.origin).href;
}

function fileTypeLabel(extension) {
    const ext = (extension || '').toLowerCase();
    const map = {
        swf: 'Flash (SWF)',
        cbz: 'Comic (CBZ)',
        cbr: 'Comic (CBR)',
        mp3: 'Audio (MP3)',
        wav: 'Audio (WAV)',
        ogg: 'Audio (OGG)',
        pdf: 'PDF',
    };
    return map[ext] || (ext ? `.${ext} file` : 'No preview');
}

/* ─── Grid items ─────────────────────────────────────────────────────────────── */
function getPreviewAspectRatio(post) {
    const w = Number(post.width);
    const h = Number(post.height);
    if (w > 0 && h > 0) return w / h;
    return 1;
}

function markMediaLoaded(mediaEl) {
    mediaEl.classList.add('loaded');
    mediaEl.closest('.masonry-media-wrap')?.classList.add('loaded');
}

function bindMediaLoaded(mediaEl) {
    const onReady = () => markMediaLoaded(mediaEl);
    if (mediaEl.tagName === 'IMG') {
        mediaEl.addEventListener('load', onReady);
        if (mediaEl.complete) onReady();
    } else {
        mediaEl.addEventListener('loadedmetadata', onReady);
        if (mediaEl.readyState >= 1) onReady();
    }
}

function createMediaWrap(post) {
    const wrap = document.createElement('div');
    wrap.className = 'masonry-media-wrap';
    wrap.style.aspectRatio = String(getPreviewAspectRatio(post));
    return wrap;
}

function removeFailedItem(div, mediaEl) {
    if (mediaEl) previewObserver.unobserve(mediaEl);
    div.remove();
}

function createGridImage(post) {
    const img = document.createElement('img');
    img.className = 'masonry-media';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.src = post.thumbnail_url;
    img.alt = post.tags || '';
    bindMediaLoaded(img);
    return img;
}

function createItem(post) {
    const div = document.createElement('div');
    div.className = 'masonry-item';
    div.dataset.id = post.id;
    const wrap = createMediaWrap(post);

    if (post.is_video) {
        const video = document.createElement('video');
        video.className = 'masonry-media';
        video.dataset.previewSrc = post.image_url;
        video.poster = post.thumbnail_url;
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = 'none';
        video.addEventListener('error', () => removeFailedItem(div, video));
        bindMediaLoaded(video);
        wrap.appendChild(video);
        previewObserver.observe(video);
    } else if (post.is_gif || (post.extension || '').toLowerCase() === 'gif') {
        // Start on the thumbnail and swap in the animation once it is on screen.
        const img = createGridImage(post);
        img.dataset.previewSrc = post.image_url;
        img.addEventListener('error', () => {
            if (img.src.endsWith(post.thumbnail_url)) {
                removeFailedItem(div, img);
            } else {
                img.src = post.thumbnail_url;
            }
        });
        wrap.appendChild(img);
        previewObserver.observe(img);
    } else {
        const img = createGridImage(post);
        img.addEventListener('error', () => removeFailedItem(div, img));
        wrap.appendChild(img);
    }

    div.appendChild(wrap);
    div.addEventListener('click', () => openModal(post));
    return div;
}

/* ─── Post loading ────────────────────────────────────────────────────────────── */
function getPageNumbers(current, total) {
    if (total <= 7) {
        return Array.from({ length: total }, (_, i) => i + 1);
    }

    const pages = new Set([1, total, current, current - 1, current + 1]);
    if (current <= 3) {
        pages.add(2);
        pages.add(3);
    }
    if (current >= total - 2) {
        pages.add(total - 1);
        pages.add(total - 2);
    }

    const sorted = [...pages].filter(p => p >= 1 && p <= total).sort((a, b) => a - b);
    const result = [];
    for (let i = 0; i < sorted.length; i++) {
        if (i > 0 && sorted[i] - sorted[i - 1] > 1) result.push('…');
        result.push(sorted[i]);
    }
    return result;
}

function openPageJump() {
    pageJumpInput.min = '1';
    pageJumpInput.max = String(totalPages);
    pageJumpInput.value = String(currentPage);
    pageJumpHint.textContent = `of ${totalPages.toLocaleString()}`;
    pageJumpEl.classList.remove('hidden');
    pageJumpInput.focus();
    pageJumpInput.select();
}

function closePageJump() {
    pageJumpEl.classList.add('hidden');
}

function submitPageJump() {
    const page = parseInt(pageJumpInput.value, 10);
    if (Number.isNaN(page) || page < 1 || page > totalPages) {
        showToast(`Enter a page between 1 and ${totalPages.toLocaleString()}`, 'error');
        pageJumpInput.focus();
        return;
    }
    closePageJump();
    if (page !== currentPage) loadPosts(page);
}

function renderPagination() {
    if (totalPages <= 1) {
        paginationEl.classList.add('hidden');
        paginationEl.innerHTML = '';
        return;
    }

    const items = getPageNumbers(currentPage, totalPages);
    const buttons = items.map(item => {
        if (item === '…') {
            return '<button type="button" class="pagination-ellipsis" aria-label="Go to page">…</button>';
        }
        const active = item === currentPage ? ' active' : '';
        return `<button class="pagination-btn${active}" data-page="${item}"${active ? ' aria-current="page"' : ''}>${item}</button>`;
    }).join('');

    paginationEl.innerHTML = `
        <button class="pagination-btn pagination-nav" data-page="${currentPage - 1}" ${currentPage === 1 ? 'disabled' : ''} aria-label="Previous page">‹</button>
        ${buttons}
        <button class="pagination-btn pagination-nav" data-page="${currentPage + 1}" ${currentPage === totalPages ? 'disabled' : ''} aria-label="Next page">›</button>
    `;
    paginationEl.classList.remove('hidden');

    paginationEl.querySelectorAll('.pagination-btn[data-page]').forEach(btn => {
        btn.addEventListener('click', () => {
            const page = parseInt(btn.dataset.page, 10);
            if (!Number.isNaN(page) && page >= 1 && page <= totalPages && page !== currentPage) {
                loadPosts(page);
            }
        });
    });

    paginationEl.querySelectorAll('.pagination-ellipsis').forEach(btn => {
        btn.addEventListener('click', openPageJump);
    });
}

async function loadPosts(page = 1) {
    if (isLoading) return;

    isLoading = true;
    currentPage = page;
    emptyState.classList.add('hidden');
    paginationEl.classList.add('hidden');
    // Drop observations for the outgoing page so they don't pin removed nodes.
    previewObserver.disconnect();
    grid.innerHTML = '';
    loadingIndicator.classList.remove('hidden');

    try {
        const res = await fetch(`/api/posts?q=${encodeURIComponent(currentQuery)}&page=${page}&limit=${LIMIT}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        totalPosts = data.total;
        totalPages = Math.max(1, Math.ceil(data.total / LIMIT));
        statsEl.textContent = `${totalPosts.toLocaleString()} posts`;

        grid.innerHTML = '';

        if (data.posts.length === 0) {
            emptyState.classList.remove('hidden');
        } else {
            const frag = document.createDocumentFragment();
            for (const post of data.posts) {
                frag.appendChild(createItem(post));
                if (post.id > maxId) maxId = post.id;
            }
            grid.appendChild(frag);
        }

        renderPagination();
        window.scrollTo(0, 0);
    } catch (e) {
        console.error('Failed to load posts', e);
        grid.innerHTML = '';
        showToast('Failed to load posts', 'error');
        statsEl.textContent = '';
        paginationEl.classList.add('hidden');
    } finally {
        isLoading = false;
        loadingIndicator.classList.add('hidden');
    }
}

/* ─── Modal ───────────────────────────────────────────────────────────────────── */
function openModal(post) {
    const imgContainer = modalImg.parentNode;
    const mediaUrl = absUrl(post.image_url);
    const thumbUrl = absUrl(post.thumbnail_url);

    // Remove any leftover placeholder or video from a previous post
    imgContainer.querySelector('.modal-no-preview')?.remove();
    const oldVideo = imgContainer.querySelector('.modal-video');
    if (oldVideo) { oldVideo.pause(); oldVideo.src = ''; oldVideo.remove(); }

    modalIdBadge.textContent = `#${post.id}`;
    modalOpenLink.href = post.image_url;
    modalOpenLink.title = post.is_video ? 'Open video' : 'Open full image';
    modalOpenLink.style.display = post.is_video ? 'none' : '';

    const showNoPreview = () => {
        modalImg.style.display = 'none';
        let ph = imgContainer.querySelector('.modal-no-preview');
        if (!ph) {
            ph = document.createElement('div');
            ph.className = 'modal-no-preview';
            imgContainer.insertBefore(ph, modalImg);
        }
        ph.innerHTML = `<span class="no-preview-type">${escapeHtml(fileTypeLabel(post.extension))}</span>` +
            `<span class="no-preview-id">#${post.id}</span>` +
            `<span class="no-preview-hint">Preview not available</span>`;
    };

    modalImg.onload = null;
    modalImg.onerror = null;

    if (post.is_video) {
        modalImg.style.display = 'none';
        modalImg.src = '';
        const video = document.createElement('video');
        video.className = 'modal-video';
        video.src = mediaUrl;
        video.controls = true;
        video.autoplay = true;
        video.loop = true;
        video.playsInline = true;
        video.onerror = () => {
            video.remove();
            showNoPreview();
        };
        imgContainer.insertBefore(video, modalImg);
    } else {
        // Show thumbnail immediately (same asset as the grid), upgrade to full file when available.
        modalImg.style.display = 'block';
        modalImg.src = thumbUrl;
        modalImg.onerror = () => {
            modalImg.onerror = null;
            showNoPreview();
        };

        const full = new Image();
        full.onload = () => {
            modalImg.src = mediaUrl;
        };
        full.src = mediaUrl;
    }

    const tagHTML = (post.tags || '')
        .split(' ')
        .filter(Boolean)
        .map(t => `<span class="tag" data-type="general">${escapeHtml(t)}</span>`)
        .join('');

    const variantHTML = (post.variant || '')
        .split(',')
        .filter(Boolean)
        .map(t => `<span class="tag tag-variant">${escapeHtml(t.trim())}</span>`)
        .join('');

    const subvariantHTML = (post.subvariant || '')
        .split(',')
        .filter(Boolean)
        .map(t => `<span class="tag tag-subvariant">${escapeHtml(t.trim())}</span>`)
        .join('');

    const date = post.date_uploaded
        ? new Date(post.date_uploaded).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
        : 'Unknown';

    const size = (post.width && post.height) ? `${post.width} × ${post.height}` : '—';

    const classificationHTML = variantHTML || subvariantHTML
        ? `${variantHTML}${subvariantHTML}`
        : '';
    const classificationLabel = variantHTML && subvariantHTML
        ? 'Classification'
        : variantHTML ? 'Variant' : 'Subvariant';

    modalMeta.innerHTML = `
        <dl class="meta-facts">
            <div class="meta-fact">
                <dt>Uploader</dt>
                <dd>${escapeHtml(post.uploader || 'Unknown')}</dd>
            </div>
            <div class="meta-fact meta-fact-inline">
                <dt>Size</dt>
                <dd>${escapeHtml(size)}</dd>
                <span class="meta-sep">·</span>
                <dt>Date</dt>
                <dd>${escapeHtml(date)}</dd>
            </div>
        </dl>
        ${classificationHTML ? `<div class="tags-section"><span class="tags-section-label">${classificationLabel}</span><div class="tags-row">${classificationHTML}</div></div>` : ''}
        ${tagHTML ? `<div class="tags-section"><span class="tags-section-label">Tags</span><div class="tags-row">${tagHTML}</div></div>` : ''}
    `;

    modalMeta.querySelectorAll('.tag').forEach(el => {
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            const tag = el.textContent;
            searchInput.value = tag;
            updateClearButton();
            currentQuery = tag;
            closeModal();
            loadPosts(1);
        });
    });

    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    modal.classList.add('hidden');
    modalImg.src = '';
    modalImg.style.display = '';
    modalImg.onerror = null;
    modalImg.onload = null;
    modalOpenLink.style.display = '';
    const video = modalImg.parentNode.querySelector('.modal-video');
    if (video) { video.pause(); video.src = ''; video.remove(); }
    document.body.style.overflow = '';
}

/* ─── Utilities ───────────────────────────────────────────────────────────────── */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function updateClearButton() {
    if (searchInput.value.length > 0) {
        searchClear.classList.add('visible');
    } else {
        searchClear.classList.remove('visible');
    }
}

/* ─── Search ──────────────────────────────────────────────────────────────────── */
let debounceTimer;

searchInput.addEventListener('input', () => {
    updateClearButton();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        updateAutocomplete();
    }, 80);
});

searchInput.addEventListener('keydown', (e) => {
    const items = autocomplete.querySelectorAll('.autocomplete-item');

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (!autocomplete.classList.contains('active') || items.length === 0) return;
        selectedIndex = (selectedIndex + 1) % items.length;
        updateSelection(items);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (!autocomplete.classList.contains('active') || items.length === 0) return;
        selectedIndex = (selectedIndex - 1 + items.length) % items.length;
        updateSelection(items);
    } else if (e.key === 'Enter') {
        clearTimeout(debounceTimer);
        if (autocomplete.classList.contains('active') && selectedIndex >= 0 && items[selectedIndex]) {
            pickItem(items[selectedIndex]);
        } else {
            currentQuery = searchInput.value.trim();
            autocomplete.classList.remove('active');
            selectedIndex = -1;
            loadPosts(1);
        }
    } else if (e.key === 'Escape') {
        autocomplete.classList.remove('active');
        selectedIndex = -1;
    }
});

function updateSelection(items) {
    items.forEach((el, i) => {
        if (i === selectedIndex) {
            el.classList.add('selected');
            el.scrollIntoView({ block: 'nearest' });
        } else {
            el.classList.remove('selected');
        }
    });
}

searchClear.addEventListener('click', () => {
    searchInput.value = '';
    updateClearButton();
    currentQuery = '';
    autocomplete.classList.remove('active');
    selectedIndex = -1;
    loadPosts(1);
    searchInput.focus();
});

/* ─── Autocomplete ────────────────────────────────────────────────────────────── */
async function updateAutocomplete() {
    const val = searchInput.value;
    const words = val.trim().split(/\s+/);
    const last = words[words.length - 1];

    // Detect prefix type (variant:, subvariant:)
    let prefixType = null;
    let searchPrefix = last;
    if (last.startsWith('variant:') && last.length > 8) {
        prefixType = 'variant';
        searchPrefix = last.substring(8);
    } else if (last.startsWith('subvariant:') && last.length > 11) {
        prefixType = 'subvariant';
        searchPrefix = last.substring(11);
    }

    // Hide autocomplete if the current word is empty
    if (!searchPrefix || searchPrefix.length < 1) {
        if (autocompleteAbortController) {
            autocompleteAbortController.abort();
            autocompleteAbortController = null;
        }
        autocomplete.classList.remove('active');
        selectedIndex = -1;
        return;
    }

    const requestId = ++autocompleteRequestId;
    const requestedPrefix = searchPrefix.toLowerCase();

    if (autocompleteAbortController) {
        autocompleteAbortController.abort();
    }
    autocompleteAbortController = new AbortController();

    try {
        const res = await fetch(`/api/tags?prefix=${encodeURIComponent(searchPrefix)}`, {
            signal: autocompleteAbortController.signal,
        });
        if (!res.ok) return;
        const data = await res.json();

        // Discard stale responses
        if (requestId !== autocompleteRequestId) return;
        const latestWords = searchInput.value.trim().split(/\s+/);
        const latestLast = latestWords[latestWords.length - 1] || '';
        let latestPrefix = null;
        let latestSearchPrefix = latestLast;
        if (latestLast.startsWith('variant:') && latestLast.length > 8) {
            latestPrefix = 'variant';
            latestSearchPrefix = latestLast.substring(8);
        } else if (latestLast.startsWith('subvariant:') && latestLast.length > 11) {
            latestPrefix = 'subvariant';
            latestSearchPrefix = latestLast.substring(11);
        }
        if (latestSearchPrefix.toLowerCase() !== requestedPrefix) return;

        // Filter results by prefix type if applicable
        let filteredTags = data.tags || [];
        if (prefixType === 'variant') {
            filteredTags = filteredTags.filter(t => t.startsWith('variant:')).map(t => t.substring(8));
        } else if (prefixType === 'subvariant') {
            filteredTags = filteredTags.filter(t => t.startsWith('subvariant:')).map(t => t.substring(11));
        } else {
            // For general search, exclude prefixed entries from showing without prefix
            filteredTags = filteredTags.filter(t => !t.startsWith('variant:') && !t.startsWith('subvariant:'));
        }

        if (!filteredTags.length) {
            autocomplete.classList.remove('active');
            selectedIndex = -1;
            return;
        }

        selectedIndex = -1;
        autocomplete.innerHTML = filteredTags
            .map(t => {
                const displayText = prefixType ? t : t;
                const prefixClass = prefixType ? `autocomplete-${prefixType}` : '';
                return `<div class="autocomplete-item ${prefixClass}" role="option" tabindex="-1">${escapeHtml(displayText)}</div>`;
            })
            .join('');
        autocomplete.classList.add('active');

        autocomplete.querySelectorAll('.autocomplete-item').forEach((el) => {
            el.addEventListener('mousedown', (e) => {
                e.preventDefault();
                pickItem(el);
            });
        });
    } catch (e) {
        if (e.name !== 'AbortError') {
            console.error('Autocomplete error:', e);
        }
    }
}

function pickItem(el) {
    clearTimeout(debounceTimer);
    const val = searchInput.value;
    const words = val.trim().split(/\s+/);
    const last = words[words.length - 1];
    const selectedTag = el.textContent;

    // Reconstruct with prefix if applicable
    let replacement = selectedTag;
    if (last.startsWith('variant:') && last.length > 8) {
        replacement = `variant:${selectedTag}`;
    } else if (last.startsWith('subvariant:') && last.length > 11) {
        replacement = `subvariant:${selectedTag}`;
    }

    words[words.length - 1] = replacement;
    searchInput.value = words.join(' ') + ' ';
    updateClearButton();
    autocomplete.classList.remove('active');
    selectedIndex = -1;
    searchInput.focus();
    currentQuery = searchInput.value.trim();
    loadPosts(1);
}

document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-container')) {
        autocomplete.classList.remove('active');
        selectedIndex = -1;
    }
});

/* ─── Modal events ────────────────────────────────────────────────────────────── */
document.querySelector('.modal-close').addEventListener('click', closeModal);
document.querySelector('.modal-backdrop').addEventListener('click', closeModal);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

pageJumpForm.addEventListener('submit', (e) => {
    e.preventDefault();
    submitPageJump();
});
pageJumpEl.querySelector('.page-jump-cancel').addEventListener('click', closePageJump);
pageJumpEl.querySelector('.page-jump-backdrop').addEventListener('click', closePageJump);
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !pageJumpEl.classList.contains('hidden')) closePageJump();
});

/* ─── Console helpers ────────────────────────────────────────────────────────── */
function addConsoleLine(message, type = 'info') {
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    line.innerHTML = `<span class="timestamp">${time}</span><span class="message">${escapeHtml(message)}</span>`;
    consoleOutput.appendChild(line);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function clearConsole() {
    consoleOutput.innerHTML = '';
}

/* ─── Scrape button ───────────────────────────────────────────────────────────── */
scrapeBtn.addEventListener('click', async () => {
    if (isScraping) {
        scrapeBtn.disabled = true;
        try {
            const res = await fetch('/api/scrape/stop', { method: 'POST' });
            const data = await res.json();
            addConsoleLine(data.message || 'Stop signal sent', 'system');
            showToast(data.message || 'Stop signal sent', 'info');
        } catch (e) {
            addConsoleLine('Failed to stop scrape: ' + e.message, 'error');
            showToast('Failed to stop scrape', 'error');
        } finally {
            scrapeBtn.disabled = false;
        }
        return;
    }

    scrapeBtn.disabled = true;
    scrapeBtnIconPlay.classList.add('hidden');
    scrapeBtnIconStop.classList.remove('hidden');
    scrapeBtnLabel.textContent = 'Stop';

    try {
        const res = await fetch('/api/scrape/start', { method: 'POST' });
        console.log('Scrape start response status:', res.status);
        const data = await res.json();
        console.log('Scrape start response data:', data);
        if (data.error) {
            addConsoleLine(data.error, 'error');
            showToast(data.error, 'error');
            resetScrapeButton();
        } else {
            addConsoleLine(data.status?.message || 'Scrape started', 'system');
        }
    } catch (e) {
        addConsoleLine('Failed to start scrape: ' + e.message, 'error');
        showToast('Failed to start scrape: ' + e.message, 'error');
        resetScrapeButton();
    } finally {
        scrapeBtn.disabled = false;
    }
});

function resetScrapeButton() {
    isScraping = false;
    scrapeBtn.classList.remove('running');
    scrapeBtnIconPlay.classList.remove('hidden');
    scrapeBtnIconStop.classList.add('hidden');
    scrapeBtnLabel.textContent = 'Scrape';
    consoleDot.classList.remove('active', 'error');
}

/* ─── Console toggle ─────────────────────────────────────────────────────────── */
consoleToggle.addEventListener('click', () => {
    const isOpen = !scrapeConsole.classList.contains('hidden');
    if (isOpen) {
        scrapeConsole.classList.add('hidden');
        consoleToggle.classList.remove('active');
    } else {
        scrapeConsole.classList.remove('hidden');
        consoleToggle.classList.add('active');
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
});

consoleClear.addEventListener('click', clearConsole);
consoleClose.addEventListener('click', () => {
    scrapeConsole.classList.add('hidden');
    consoleToggle.classList.remove('active');
});

/* ─── Scrape UI helpers ────────────────────────────────────────────────────────── */
function setScrapeRunning(running) {
    isScraping = running;
    if (running) {
        scrapeBtn.classList.add('running');
        scrapeBtnIconPlay.classList.add('hidden');
        scrapeBtnIconStop.classList.remove('hidden');
        scrapeBtnLabel.textContent = 'Stop';
        consoleDot.classList.add('active');
        consoleDot.classList.remove('error');
    } else {
        resetScrapeButton();
    }
}

/* ─── SSE ─────────────────────────────────────────────────────────────────────── */
function connectSSE() {
    const es = new EventSource('/api/events');

    es.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
            case 'post_start':
                addConsoleLine(`Scraping post #${data.data.id}...`, 'info');
                break;
            case 'console':
                addConsoleLine(data.data.message, data.data.level || 'info');
                break;
            case 'post_done':
                if (data.data.status === 'completed') {
                    addConsoleLine(`Post #${data.data.id} completed`, 'success');
                    handleNewPost(data.data.id);
                } else if (data.data.status === 'failed') {
                    addConsoleLine(`Post #${data.data.id} failed`, 'error');
                } else if (data.data.status === 'empty') {
                    addConsoleLine(`Post #${data.data.id} empty (404)`, 'warning');
                }
                break;
            case 'status':
                addConsoleLine(data.data.message, 'system');
                setScrapeRunning(true);
                break;
            case 'complete': {
                setScrapeRunning(false);
                const s = data.data.stats;
                const msg = `Done — ${s.completed} saved, ${s.skipped} skipped, ${s.empty} empty, ${s.failed} failed`;
                addConsoleLine(msg, 'success');
                showToast(msg, 'success', 6000);
                break;
            }
            case 'error':
                setScrapeRunning(false);
                addConsoleLine(`Error: ${data.data.message}`, 'error');
                showToast(`Scrape error: ${data.data.message}`, 'error', 6000);
                consoleDot.classList.add('error');
                break;
        }
    };

    es.onerror = () => {
        console.log('SSE lost, reconnecting in 3s…');
        es.close();
        setTimeout(connectSSE, 3000);
    };

    return es;
}

/* ─── New post (live update) ──────────────────────────────────────────────────── */
async function handleNewPost(postId) {
    try {
        const res = await fetch(`/api/recent?after_id=${postId - 1}`);
        const data = await res.json();
        const post = data.posts.find(p => p.id === postId);
        if (post) {
            if (currentQuery === '' && currentPage === 1 && window.scrollY < 200) {
                const item = createItem(post);
                item.style.animation = 'fadeIn 0.35s cubic-bezier(0.4,0,0.2,1)';
                grid.insertBefore(item, grid.firstChild);
            }
            if (post.id > maxId) maxId = post.id;
        }
    } catch (e) {
        console.error('Failed to fetch new post', e);
    }
}

/* ─── Scrape status poll ──────────────────────────────────────────────────────── */
/* Progress arrives over SSE; polling only covers a scrape started elsewhere
   (another tab, the CLI) or an SSE drop, so it idles when nothing is running. */
let statusPollTimer = null;

async function checkScrapeStatus() {
    try {
        const res = await fetch('/api/scrape/status');
        const data = await res.json();
        setScrapeRunning(Boolean(data.running));
    } catch {
        // silently ignore poll errors
    }
    scheduleStatusPoll();
}

function scheduleStatusPoll() {
    clearTimeout(statusPollTimer);
    if (document.hidden) return;
    statusPollTimer = setTimeout(checkScrapeStatus, isScraping ? 2000 : 15000);
}

document.addEventListener('visibilitychange', () => {
    if (document.hidden) clearTimeout(statusPollTimer);
    else checkScrapeStatus();
});

/* ─── Init ────────────────────────────────────────────────────────────────────── */
loadPosts(1);
connectSSE();
checkScrapeStatus();
