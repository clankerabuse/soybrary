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
const sentinel       = document.getElementById('sentinel');
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
let isLoading    = false;
let hasMore      = true;
let maxId        = 0;
let isScraping   = false;
let selectedIndex = -1;
let autocompleteAbortController = null;
let autocompleteRequestId = 0;

const LIMIT = 50;

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
function createItem(post) {
    const div = document.createElement('div');
    div.className = 'masonry-item';
    div.dataset.id = post.id;
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = post.thumbnail_url;
    img.alt = post.tags || '';
    img.addEventListener('error', () => {
        img.style.display = 'none';
        const placeholder = document.createElement('div');
        placeholder.className = 'no-preview';
        const label = fileTypeLabel(post.extension);
        placeholder.innerHTML = `<span class="no-preview-type">${escapeHtml(label)}</span><span class="no-preview-id">#${post.id}</span>`;
        div.appendChild(placeholder);
    });
    div.appendChild(img);
    div.addEventListener('click', () => openModal(post));
    return div;
}

/* ─── Post loading ────────────────────────────────────────────────────────────── */
async function loadPosts(reset = false) {
    if (isLoading) return;
    if (!reset && !hasMore) return;

    isLoading = true;
    emptyState.classList.add('hidden');

    if (reset) {
        grid.innerHTML = '';
        currentPage = 1;
        hasMore = true;
        maxId = 0;
        showSkeletons(20);
    }

    if (!reset) {
        loadingIndicator.classList.remove('hidden');
    }

    try {
        const res = await fetch(`/api/posts?q=${encodeURIComponent(currentQuery)}&page=${currentPage}&limit=${LIMIT}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        statsEl.textContent = `${data.total.toLocaleString()} posts`;

        clearSkeletons();

        if (data.posts.length === 0) {
            hasMore = false;
            if (currentPage === 1) {
                emptyState.classList.remove('hidden');
            }
        } else {
            const frag = document.createDocumentFragment();
            for (const post of data.posts) {
                frag.appendChild(createItem(post));
                if (post.id > maxId) maxId = post.id;
            }
            grid.appendChild(frag);
            currentPage++;
            if (data.posts.length < LIMIT) hasMore = false;
        }
    } catch (e) {
        console.error('Failed to load posts', e);
        clearSkeletons();
        showToast('Failed to load posts', 'error');
        statsEl.textContent = '';
    } finally {
        isLoading = false;
        loadingIndicator.classList.add('hidden');
    }
}

/* ─── Modal ───────────────────────────────────────────────────────────────────── */
function openModal(post) {
    const imgContainer = modalImg.parentNode;

    // Remove any leftover placeholder or video from a previous post
    imgContainer.querySelector('.modal-no-preview')?.remove();
    const oldVideo = imgContainer.querySelector('.modal-video');
    if (oldVideo) { oldVideo.pause(); oldVideo.src = ''; oldVideo.remove(); }

    modalIdBadge.textContent = `#${post.id}`;
    modalOpenLink.href = post.image_url;
    modalOpenLink.title = post.is_video ? 'Open video' : 'Open full image';
    modalOpenLink.style.display = post.is_video ? 'none' : '';

    if (post.is_video) {
        // Hide the img element and show a <video> player instead
        modalImg.style.display = 'none';
        modalImg.src = '';
        const video = document.createElement('video');
        video.className = 'modal-video';
        video.src = post.image_url;
        video.controls = true;
        video.autoplay = true;
        video.loop = true;
        video.playsInline = true;
        imgContainer.insertBefore(video, modalImg);
    } else {
        modalImg.style.display = '';
        modalImg.src = post.image_url;
        // If the file can't be rendered as an image, show a typed placeholder
        modalImg.onerror = () => {
            modalImg.style.display = 'none';
            let ph = imgContainer.querySelector('.modal-no-preview');
            if (!ph) {
                ph = document.createElement('div');
                ph.className = 'modal-no-preview';
                imgContainer.insertBefore(ph, modalImg);
            }
            ph.innerHTML = `<span class="no-preview-type">${escapeHtml(fileTypeLabel(post.extension))}</span>` +
                `<span class="no-preview-id">#${post.id}</span>` +
                `<a class="no-preview-download" href="${post.image_url}" download>Download file</a>`;
        };
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

    modalMeta.innerHTML = `
        <div class="meta-grid">
            <div class="meta-field">
                <span class="meta-label">Uploader</span>
                <span class="meta-value">${escapeHtml(post.uploader || 'Unknown')}</span>
            </div>
            <div class="meta-field">
                <span class="meta-label">Dimensions</span>
                <span class="meta-value">${escapeHtml(size)}</span>
            </div>
            <div class="meta-field">
                <span class="meta-label">Date</span>
                <span class="meta-value">${escapeHtml(date)}</span>
            </div>
        </div>
        ${variantHTML ? `<div class="tags-section"><span class="tags-section-label">Variant</span><div class="tags-row">${variantHTML}</div></div>` : ''}
        ${subvariantHTML ? `<div class="tags-section"><span class="tags-section-label">Subvariant</span><div class="tags-row">${subvariantHTML}</div></div>` : ''}
        ${tagHTML ? `<div class="tags-section"><span class="tags-section-label">Tags</span><div class="tags-row">${tagHTML}</div></div>` : ''}
    `;

    modalMeta.querySelectorAll('.tag').forEach(el => {
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            const tag = el.textContent;
            const type = el.dataset.type || el.classList.contains('tag-variant') ? 'variant' : el.classList.contains('tag-subvariant') ? 'subvariant' : 'general';
            let searchVal = tag;
            if (type === 'variant') searchVal = `variant:${tag}`;
            else if (type === 'subvariant') searchVal = `subvariant:${tag}`;
            searchInput.value = searchVal;
            updateClearButton();
            currentQuery = searchVal;
            closeModal();
            loadPosts(true);
        });
    });

    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    modal.classList.add('hidden');
    modalImg.src = '';
    modalImg.style.display = '';
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
            loadPosts(true);
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
    loadPosts(true);
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
    loadPosts(true);
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

/* ─── Infinite scroll ─────────────────────────────────────────────────────────── */
const observer = new IntersectionObserver(
    (entries) => {
        if (entries[0].isIntersecting && hasMore && !isLoading) loadPosts();
    },
    { rootMargin: '400px' }
);
observer.observe(sentinel);

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
            if (currentQuery === '' && window.scrollY < 200) {
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
async function checkScrapeStatus() {
    try {
        const res = await fetch('/api/scrape/status');
        const data = await res.json();
        if (data.running) {
            setScrapeRunning(true);
            setScrapeStatus(data.message || `Scraping ID ${data.current_id}…`);
        } else if (isScraping) {
            setScrapeRunning(false);
        }
    } catch {
        // silently ignore poll errors
    }
}

/* ─── Init ────────────────────────────────────────────────────────────────────── */
loadPosts(true);
connectSSE();
setInterval(checkScrapeStatus, 2000);
