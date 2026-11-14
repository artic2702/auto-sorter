/*
 * Auto-Sorter Desktop — Renderer Application.
 *
 * Zero external frameworks, zero CDN assets, strict CSP compatible (no inline
 * event attributes). Styled on the PalmPay light emerald design system.
 */
(function () {
  'use strict';

  const api = window.autosorter;
  const icon = window.AutoSorterIcons;

  const PREFS_KEY = 'autosorter.prefs.v1';
  const GRID_SIZES = {
    compact: 118,
    comfortable: 158,
    large: 210
  };

  const STAGE_LABELS = [
    { num: 1, short: 'Scan & Hash' },
    { num: 2, short: 'Detect Faces' },
    { num: 3, short: 'Faceprints' },
    { num: 4, short: 'Identify People' }
  ];

  // ---------------------------------------------------------------- State

  function loadPrefs() {
    try {
      const raw = localStorage.getItem(PREFS_KEY);
      if (!raw) return {};
      return JSON.parse(raw) || {};
    } catch (_err) {
      return {};
    }
  }

  const savedPrefs = loadPrefs();

  const state = {
    sysInfo: null,
    bridgeConnected: true,

    photoFolder: null,
    customDataFolder: null,
    dataFolder: null,
    useCustomDataOnWelcome: false,

    recentLibraries: Array.isArray(savedPrefs.recentLibraries) ? savedPrefs.recentLibraries : [],
    speedMode: ['quiet', 'normal', 'fast'].includes(savedPrefs.speedMode) ? savedPrefs.speedMode : 'normal',
    photoGridSize: ['compact', 'comfortable', 'large'].includes(savedPrefs.photoGridSize)
      ? savedPrefs.photoGridSize
      : 'comfortable',
    lastExportFolder: savedPrefs.lastExportFolder || '',

    tab: 'overview', // 'overview' | 'people' | 'photos' | 'albums' | 'settings'

    library: null,

    // People tab & Person Detail state
    people: [],
    peopleSearch: '',
    peopleSort: 'most_photos',
    inlineRenameId: null,
    selectedPersonId: null,
    personDetail: null,
    selectedFaceIds: new Set(),
    lastClickedFaceIndex: null,

    // Virtualized Photos tab state
    photos: [],
    photosFilter: 'all',
    photosPersonId: '',
    photosSearch: '',
    photosLoading: false,

    // Smart Albums tab state
    albumsSummary: [],
    selectedAlbumType: null,
    albumDetail: null,

    // Live pipeline state
    pipeline: {
      running: false,
      cancelling: false,
      stage_num: 0,
      total_stages: 4,
      stage_name: '',
      done: 0,
      total: 0,
      detail: '',
      rss_mb: 0,
      lastSummary: null,
      lastError: null
    },

    // Live export state
    exportJob: {
      running: false,
      cancelling: false,
      done: 0,
      total: 0,
      detail: '',
      exportRoot: '',
      lastResult: null,
      lastError: null
    },

    // Active modal descriptor
    modal: null,

    // Interactive photo viewer / lightbox state
    lightbox: {
      open: false,
      loading: false,
      imageId: null,
      photo: null,
      faces: [],
      selectedFaceId: null,
      showBoxes: true,
      showSidebar: true,
      contextList: [], // [{ imageId, faceId? }]
      contextIndex: -1,
      similarResult: null,
      similarLoading: false
    }
  };

  // Virtual grid controller cleanup
  let vgridCleanup = null;
  // Lightbox image ResizeObserver cleanup
  let lbResizeObserver = null;
  // Search debounce timer
  let photoSearchTimer = null;

  // DOM anchors
  const dom = {
    brandBtn: document.getElementById('brandBtn'),
    brandMark: document.getElementById('brandMark'),
    topNav: document.getElementById('topNav'),
    topbarRight: document.getElementById('topbarRight'),
    workspace: document.getElementById('workspace'),
    view: document.getElementById('view'),
    modalHost: document.getElementById('modalHost'),
    lightboxHost: document.getElementById('lightboxHost'),
    dropOverlay: document.getElementById('dropOverlay'),
    dropOverlayIcon: document.getElementById('dropOverlayIcon'),
    toastStack: document.getElementById('toastStack')
  };

  // ---------------------------------------------------------------- Helpers

  function savePrefs() {
    try {
      localStorage.setItem(
        PREFS_KEY,
        JSON.stringify({
          recentLibraries: state.recentLibraries.slice(0, 8),
          speedMode: state.speedMode,
          photoGridSize: state.photoGridSize,
          lastExportFolder: state.lastExportFolder
        })
      );
    } catch (_err) {
      // ignore storage quota issues
    }
  }

  function esc(val) {
    if (val === null || val === undefined) return '';
    return String(val)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmtNum(n) {
    const num = Number(n || 0);
    return Number.isFinite(num) ? num.toLocaleString() : '0';
  }

  function fmtBytes(bytes) {
    const b = Number(bytes || 0);
    if (!b || b < 0) return '0 B';
    if (b < 1024) return `${b} B`;
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
    if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)} MB`;
    return `${(b / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }

  function baseName(p) {
    if (!p) return '';
    const trimmed = String(p).replace(/[\\/]+$/, '');
    const parts = trimmed.split(/[\\/]/);
    return parts[parts.length - 1] || trimmed;
  }

  function mediaSrc(filePath, bust = '') {
    if (!filePath) return '';
    return api.mediaUrl(filePath, bust);
  }

  function folderParams(extra = {}) {
    return {
      photo_folder: state.photoFolder,
      data_folder: state.customDataFolder || state.dataFolder || undefined,
      ...extra
    };
  }

  function toast(message, kind = 'default', durationMs = 3400) {
    const el = document.createElement('div');
    el.className = `toast ${kind === 'ok' ? 'ok' : kind === 'bad' ? 'bad' : ''}`;
    const ic = kind === 'ok' ? icon('check') : kind === 'bad' ? icon('close') : icon('sparkle');
    el.innerHTML = `<span style="width:16px;height:16px;display:grid;place-items:center;flex-shrink:0">${ic}</span><span>${esc(message)}</span>`;
    dom.toastStack.appendChild(el);
    setTimeout(() => {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, durationMs);
  }

  async function copyText(text, label = 'Path') {
    try {
      await navigator.clipboard.writeText(String(text || ''));
      toast(`${label} copied to clipboard`, 'ok', 2200);
    } catch (_err) {
      toast('Could not access clipboard', 'bad');
    }
  }

  function recordRecentLibrary(photoFolder, customDataFolder) {
    if (!photoFolder) return;
    const norm = String(photoFolder);
    const filtered = state.recentLibraries.filter(
      (item) => String(item.photoFolder).toLowerCase() !== norm.toLowerCase()
    );
    filtered.unshift({
      photoFolder: norm,
      customDataFolder: customDataFolder || null,
      openedAt: new Date().toISOString()
    });
    state.recentLibraries = filtered.slice(0, 8);
    savePrefs();
  }

  // ---------------------------------------------------------------- Data Loading

  async function openLibrary(photoFolder, customDataFolder = null) {
    if (!photoFolder) return;
    try {
      const info = await api.rpc('library.inspect', {
        photo_folder: photoFolder,
        data_folder: customDataFolder || undefined
      });
      if (!info.folder_exists) {
        toast('Selected folder no longer exists on disk', 'bad');
        return;
      }
      state.photoFolder = info.photo_folder || photoFolder;
      state.customDataFolder = customDataFolder || null;
      state.dataFolder = info.data_folder;
      state.library = info;
      state.selectedPersonId = null;
      state.personDetail = null;
      state.selectedAlbumType = null;
      state.albumDetail = null;
      state.selectedFaceIds.clear();
      state.photosPersonId = '';
      state.photosFilter = 'all';
      state.photosSearch = '';
      state.pipeline.lastSummary = null;
      state.pipeline.lastError = null;
      state.tab = 'overview';

      recordRecentLibrary(state.photoFolder, state.customDataFolder);
      await refreshAllLibraryCollections();
      renderApp();
    } catch (err) {
      toast(err.message || 'Could not inspect folder', 'bad');
    }
  }

  async function refreshLibraryInspect() {
    if (!state.photoFolder) return;
    const info = await api.rpc('library.inspect', folderParams());
    state.library = info;
    state.dataFolder = info.data_folder;
  }

  async function refreshPeopleList() {
    if (!state.photoFolder) return;
    const res = await api.rpc('people.list', folderParams());
    state.people = res.persons || [];
  }

  async function refreshAlbumsSummary() {
    if (!state.photoFolder) return;
    const res = await api.rpc('albums.summary', folderParams());
    state.albumsSummary = res.albums || [];
  }

  async function refreshPhotosList() {
    if (!state.photoFolder) return;
    state.photosLoading = true;
    try {
      const res = await api.rpc(
        'photos.list',
        folderParams({
          filter: state.photosFilter,
          person_id: state.photosPersonId || null,
          search: state.photosSearch || ''
        })
      );
      state.photos = res.photos || [];
    } finally {
      state.photosLoading = false;
    }
  }

  async function refreshPersonDetail(personId) {
    if (!state.photoFolder || !personId) return;
    try {
      const res = await api.rpc('people.get', folderParams({ person_id: personId }));
      state.personDetail = res;
    } catch (_err) {
      state.selectedPersonId = null;
      state.personDetail = null;
    }
  }

  async function refreshAlbumDetail(albumType) {
    if (!state.photoFolder || !albumType) return;
    const res = await api.rpc('albums.items', folderParams({ album_type: albumType }));
    state.albumDetail = res;
  }

  async function refreshAllLibraryCollections() {
    if (!state.photoFolder) return;
    await Promise.all([
      refreshLibraryInspect(),
      refreshPeopleList(),
      refreshAlbumsSummary()
    ]);
    if (state.tab === 'people' && state.selectedPersonId) {
      await refreshPersonDetail(state.selectedPersonId);
    }
    if (state.tab === 'photos') {
      await refreshPhotosList();
    }
    if (state.tab === 'albums' && state.selectedAlbumType) {
      await refreshAlbumDetail(state.selectedAlbumType);
    }
  }

  // ---------------------------------------------------------------- Pipeline & Export Actions

  async function startPipeline() {
    if (!state.photoFolder) {
      toast('Select a photo folder first', 'bad');
      return;
    }
    if (state.pipeline.running) return;

    try {
      state.pipeline.lastError = null;
      state.pipeline.lastSummary = null;
      const snap = await api.rpc(
        'pipeline.start',
        folderParams({ speed_mode: state.speedMode })
      );
      state.pipeline.running = true;
      state.pipeline.cancelling = false;
      state.pipeline.stage_num = snap.stage_num || 1;
      state.pipeline.stage_name = snap.stage_name || 'Scanning folder';
      state.pipeline.done = snap.done || 0;
      state.pipeline.total = snap.total || 0;
      state.pipeline.detail = snap.detail || 'Starting pipeline…';
      state.pipeline.rss_mb = snap.rss_mb || 0;
      renderTopbar();
      if (state.tab === 'overview') {
        renderView();
      }
    } catch (err) {
      toast(err.message || 'Could not start scan', 'bad');
    }
  }

  async function cancelPipeline() {
    if (!state.pipeline.running) return;
    try {
      state.pipeline.cancelling = true;
      updatePipelineDomInPlace();
      await api.rpc('pipeline.cancel', {});
      toast('Stopping scan gracefully — progress is saved', 'default');
    } catch (err) {
      toast(err.message || 'Could not cancel scan', 'bad');
    }
  }

  // ---------------------------------------------------------------- Shell & Topbar Rendering

  function destroyVGrid() {
    if (typeof vgridCleanup === 'function') {
      vgridCleanup();
      vgridCleanup = null;
    }
  }

  function renderApp() {
    renderTopbar();
    renderView();
  }

  function renderTopbar() {
    if (!state.photoFolder) {
      dom.topNav.innerHTML = '';
      dom.topbarRight.innerHTML = state.sysInfo
        ? `<span class="muted" style="font-size:12.5px">v${esc(state.sysInfo.app_version)} · ${fmtNum(state.sysInfo.cpu_count)} CPU cores</span>`
        : '';
      return;
    }

    const st = (state.library && state.library.state) || {};
    const albums = (state.library && state.library.albums) || {};
    const albumTotal = (albums.group || 0) + (albums.unknown || 0) + (albums.docs || 0);

    const navItems = [
      { id: 'overview', label: 'Overview', icon: 'sparkle', count: null },
      { id: 'people', label: 'People', icon: 'people', count: st.persons || 0 },
      { id: 'photos', label: 'Photos', icon: 'photo', count: st.processed || 0 },
      { id: 'albums', label: 'Albums & Export', icon: 'albums', count: albumTotal },
      { id: 'settings', label: 'Settings', icon: 'settings', count: null }
    ];

    dom.topNav.innerHTML = navItems
      .map((item) => {
        const active = state.tab === item.id ? 'on' : '';
        const badge =
          item.count !== null && item.count > 0
            ? `<span class="nav-count">${fmtNum(item.count)}</span>`
            : '';
        return `
          <button class="nav-btn ${active}" type="button" data-action="switch-tab" data-tab="${item.id}">
            ${icon(item.icon)}
            <span>${esc(item.label)}</span>
            ${badge}
          </button>
        `;
      })
      .join('');

    dom.topbarRight.innerHTML = buildTopbarRightHtml();
  }

  function buildTopbarRightHtml() {
    if (!state.photoFolder) return '';
    const folderLabel = baseName(state.photoFolder);

    let scanActionHtml = '';
    if (state.pipeline.running) {
      const pct =
        state.pipeline.total > 0
          ? `${Math.min(100, Math.round((state.pipeline.done / state.pipeline.total) * 100))}%`
          : `Stage ${state.pipeline.stage_num || 1}/4`;
      scanActionHtml = `
        <span class="pill ok tabular" id="topbarPipePill" title="${esc(state.pipeline.detail)}">
          <span class="dot"></span>
          <span id="topbarPipeLabel">${esc(state.pipeline.stage_name || 'Scanning')} · ${esc(pct)}</span>
        </span>
        <button class="btn danger sm" type="button" data-action="pipeline-cancel" ${state.pipeline.cancelling ? 'disabled' : ''}>
          ${icon('stop')}
          <span>${state.pipeline.cancelling ? 'Stopping…' : 'Stop'}</span>
        </button>
      `;
    } else {
      const actionType = (state.library && state.library.action) || 'start';
      const btnText =
        actionType === 'resume' ? 'Resume Scan' : actionType === 'rescan' ? 'Re-scan' : 'Start Scan';
      const btnClass = actionType === 'rescan' ? 'btn soft sm' : 'btn sm';
      scanActionHtml = `
        <button class="${btnClass}" type="button" data-action="pipeline-start">
          ${icon(actionType === 'rescan' ? 'refresh' : 'play')}
          <span>${esc(btnText)}</span>
        </button>
      `;
    }

    return `
      <button class="folder-chip" type="button" data-action="open-switch-library" title="${esc(state.photoFolder)}">
        ${icon('folder')}
        <span class="truncate">${esc(folderLabel)}</span>
      </button>
      <span id="topbarScanArea" class="btn-row">${scanActionHtml}</span>
    `;
  }

  // ---------------------------------------------------------------- View Router

  function renderView() {
    destroyVGrid();

    if (!state.photoFolder) {
      dom.workspace.classList.remove('fixed-pane');
      dom.view.innerHTML = renderWelcomeView();
      return;
    }

    if (state.tab === 'photos') {
      dom.workspace.classList.add('fixed-pane');
      dom.view.className = 'page full-height';
      dom.view.innerHTML = renderPhotosView();
      mountVirtualPhotosGrid();
      return;
    }

    dom.workspace.classList.remove('fixed-pane');
    dom.view.className = 'page';

    switch (state.tab) {
      case 'overview':
        dom.view.innerHTML = renderOverviewView();
        break;
      case 'people':
        dom.view.innerHTML = state.selectedPersonId && state.personDetail
          ? renderPersonDetailView()
          : renderPeopleView();
        focusInlineRenameIfAny();
        break;
      case 'albums':
        dom.view.innerHTML = state.selectedAlbumType && state.albumDetail
          ? renderAlbumDetailView()
          : renderAlbumsView();
        break;
      case 'settings':
        dom.view.innerHTML = renderSettingsView();
        break;
      default:
        dom.view.innerHTML = renderOverviewView();
        break;
    }
  }

  // ---------------------------------------------------------------- 1. Welcome / Split Onboarding View

  function renderWelcomeView() {
    dom.view.className = '';
    const workers = (state.sysInfo && state.sysInfo.speed_workers) || { quiet: 1, normal: 2, fast: 4 };
    const recentHtml = state.recentLibraries.length
      ? `
        <div class="section-title" style="margin-top:22px">Recent Libraries</div>
        <div class="recent-list">
          ${state.recentLibraries
            .map(
              (item, idx) => `
              <div class="recent-item" data-action="open-recent" data-idx="${idx}" title="${esc(item.photoFolder)}">
                ${icon('folderOpen')}
                <div class="ri-text">
                  <div class="ri-name truncate">${esc(baseName(item.photoFolder))}</div>
                  <div class="ri-path truncate">${esc(item.photoFolder)}</div>
                </div>
                <button class="icon-btn" type="button" data-action="remove-recent" data-idx="${idx}" title="Remove from recent">
                  ${icon('close')}
                </button>
              </div>
            `
            )
            .join('')}
        </div>
      `
      : '';

    const customDataHtml = state.useCustomDataOnWelcome
      ? `
        <div class="field" style="margin-top:14px">
          <label>Custom Index &amp; Cache Folder (.autosorter)</label>
          <div class="btn-row" style="flex-wrap:nowrap">
            <input type="text" id="welcomeCustomDataInput" readonly value="${esc(state.customDataFolder || '')}" placeholder="Default: <photo-folder>\\.autosorter" />
            <button class="btn ghost" type="button" data-action="pick-welcome-data-folder">Browse…</button>
          </div>
          <div class="hint">Useful when scanning read-only drives, SD cards, or NAS shares.</div>
        </div>
      `
      : `
        <div style="margin-top:12px;display:flex;justify-content:space-between;align-items:center">
          <span class="muted" style="font-size:12.5px">Index stored locally in <code class="mono">.autosorter</code></span>
          <button class="btn ghost sm" type="button" data-action="toggle-welcome-custom-data">Custom index path…</button>
        </div>
      `;

    return `
      <div class="welcome-shell">
        <aside class="welcome-side">
          <div class="eyebrow">
            <span class="dot"></span>
            <span>100% On-Device Computer Vision</span>
          </div>

          <div class="welcome-hero">
            <h1>Organize thousands of photos by the people in them.</h1>
            <p class="lead">
              Auto-Sorter detects faces with YuNet, builds 128-D SFace embeddings, and clusters identities automatically — without uploading a single pixel to the cloud.
            </p>

            <div class="welcome-points">
              <div class="pt">
                <div class="pt-icon">${icon('shield')}</div>
                <div>
                  <strong>Non-destructive by design</strong><br/>
                  Your source photos are never renamed, altered, or moved. Exports always create clean copies.
                </div>
              </div>
              <div class="pt">
                <div class="pt-icon">${icon('people')}</div>
                <div>
                  <strong>Active-learning face clusters</strong><br/>
                  Rename, merge, split, or re-check albums — manual corrections lock in and refine future scans.
                </div>
              </div>
              <div class="pt">
                <div class="pt-icon">${icon('bolt')}</div>
                <div>
                  <strong>Resumable 4-stage pipeline</strong><br/>
                  Pause or close at any moment and pick up right where you left off. Supports JPEG, PNG, WebP, HEIC &amp; RAW.
                </div>
              </div>
            </div>
          </div>

          <div class="welcome-foot">
            YuNet + SFace ONNX Runtime · SQLite WAL Index · Offline Desktop Engine
          </div>
        </aside>

        <section class="welcome-main">
          <div class="welcome-card">
            <h2>Open a photo library</h2>
            <p>Choose or drop any folder containing your photos to inspect or scan.</p>

            <div class="dropzone" id="welcomeDropzone" data-action="pick-photo-folder">
              <div class="dz-badge">${icon('folderOpen')}</div>
              <strong>Drop a photo folder here</strong>
              <span>or browse your computer to get started</span>
              <button class="btn lg" type="button" data-action="pick-photo-folder">
                ${icon('folder')}
                <span>Select Photo Folder…</span>
              </button>
            </div>

            ${customDataHtml}

            <div style="margin-top:16px;display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
              <span class="muted" style="font-size:12.8px;font-weight:550">Scan speed preset</span>
              <div class="seg" role="group" aria-label="Speed preset">
                <button type="button" class="${state.speedMode === 'quiet' ? 'on' : ''}" data-action="set-speed" data-speed="quiet" title="${workers.quiet} worker thread">Quiet</button>
                <button type="button" class="${state.speedMode === 'normal' ? 'on' : ''}" data-action="set-speed" data-speed="normal" title="${workers.normal} worker threads">Normal</button>
                <button type="button" class="${state.speedMode === 'fast' ? 'on' : ''}" data-action="set-speed" data-speed="fast" title="${workers.fast} worker threads">Fast</button>
              </div>
            </div>

            ${recentHtml}
          </div>
        </section>
      </div>
    `;
  }

  // ---------------------------------------------------------------- 2. Overview Tab

  function renderOverviewView() {
    const lib = state.library || {};
    const st = lib.state || {};
    const albums = lib.albums || {};
    const albumTotal = (albums.group || 0) + (albums.unknown || 0) + (albums.docs || 0);
    const workers = (state.sysInfo && state.sysInfo.speed_workers) || { quiet: 1, normal: 2, fast: 4 };

    let bannerNote = '';
    if (state.pipeline.lastError) {
      bannerNote = `<div class="note bad"><strong>Pipeline error:</strong> ${esc(state.pipeline.lastError)}</div>`;
    } else if (state.pipeline.lastSummary) {
      const s = state.pipeline.lastSummary;
      if (s.cancelled) {
        bannerNote = `<div class="note warn"><strong>Scan paused.</strong> Progress has been saved safely to SQLite — click <strong>Resume Scan</strong> whenever you are ready.</div>`;
      } else {
        bannerNote = `<div class="note ok"><strong>Scan complete.</strong> Verified ${fmtNum(s.files_found)} files, detected ${fmtNum(s.faces_detected)} new faces, and organized ${fmtNum(s.persons_found ?? s.persons_created ?? 0)} people.</div>`;
      }
    } else if (!lib.has_db && lib.probe) {
      const countLabel = lib.probe.capped ? `${fmtNum(lib.probe.count)}+` : fmtNum(lib.probe.count);
      bannerNote = `<div class="note info"><strong>Ready to index.</strong> Found ${countLabel} supported image files in this folder. Original files will never be modified.</div>`;
    } else if (lib.action === 'resume' && !state.pipeline.running) {
      bannerNote = `<div class="note warn"><strong>Unfinished scan detected.</strong> ${fmtNum(st.pending + st.needs_detection)} photos still need processing — click <strong>Resume Scan</strong> to continue.</div>`;
    }

    const topPeople = state.people.slice(0, 6);
    const topPeopleHtml = topPeople.length
      ? `
        <div class="mini-people-list">
          ${topPeople
            .map(
              (p) => `
              <div class="mini-person" data-action="open-person" data-person-id="${p.id}">
                ${
                  p.cover_thumb_path
                    ? `<img src="${esc(mediaSrc(p.cover_thumb_path))}" alt="${esc(p.display_name)}" loading="lazy" />`
                    : `<div class="mp-fallback">${icon('person')}</div>`
                }
                <div style="min-width:0;flex:1">
                  <div class="truncate" style="font-weight:600;font-size:13.5px">${esc(p.display_name)}</div>
                  <div class="muted tabular" style="font-size:12px">${fmtNum(p.photo_count)} photo${p.photo_count === 1 ? '' : 's'}</div>
                </div>
                ${p.name_is_user_set ? `<span class="pill ok" style="padding:1px 7px;font-size:11px">Named</span>` : ''}
              </div>
            `
            )
            .join('')}
        </div>
      `
      : `<div class="empty" style="padding:26px 12px"><strong>No people clustered yet</strong>Run a scan to detect faces and group recurring people automatically.</div>`;

    return `
      <div class="page-head">
        <div>
          <h1>${esc(baseName(state.photoFolder))}</h1>
          <p class="mono" style="font-size:12.5px">${esc(state.photoFolder)}</p>
        </div>
        <div class="spacer"></div>
        <div class="btn-row">
          <button class="btn ghost sm" type="button" data-action="open-in-explorer" data-path="${esc(state.photoFolder)}">
            ${icon('external')}
            <span>Open Folder</span>
          </button>
          <button class="btn ghost sm" type="button" data-action="open-switch-library">
            ${icon('folder')}
            <span>Switch Library…</span>
          </button>
          <button class="btn soft sm" type="button" data-action="open-export-modal" ${!lib.has_db ? 'disabled' : ''}>
            ${icon('export')}
            <span>Export Sorted Copies…</span>
          </button>
        </div>
      </div>

      ${bannerNote}

      ${renderPipelineCardHtml(workers)}

      <div class="grid cols-4" style="margin-bottom:16px">
        <div class="stat clickable" data-action="switch-tab" data-tab="photos">
          <div class="label"><span>Photos Indexed</span>${icon('photo')}</div>
          <div class="value">${fmtNum(st.processed)}</div>
          <div class="foot">${st.error > 0 ? `${fmtNum(st.error)} unreadable · ` : ''}${fmtNum(st.total)} discovered</div>
        </div>

        <div class="stat clickable" data-action="jump-photos-filter" data-filter="with_faces">
          <div class="label"><span>Faces Detected</span>${icon('target')}</div>
          <div class="value">${fmtNum(st.faces)}</div>
          <div class="foot">${fmtNum(st.embedded)} 128-D faceprints</div>
        </div>

        <div class="stat accent clickable" data-action="switch-tab" data-tab="people">
          <div class="label"><span>People Found</span>${icon('people')}</div>
          <div class="value">${fmtNum(st.persons)}</div>
          <div class="foot">${fmtNum(st.named_persons)} named · ${fmtNum(Math.max(0, (st.persons || 0) - (st.named_persons || 0)))} auto-labeled</div>
        </div>

        <div class="stat clickable" data-action="switch-tab" data-tab="albums">
          <div class="label"><span>Smart Albums</span>${icon('albums')}</div>
          <div class="value">${fmtNum(albumTotal)}</div>
          <div class="foot">${fmtNum(albums.group || 0)} groups · ${fmtNum(albums.unknown || 0)} unknown · ${fmtNum(albums.docs || 0)} docs</div>
        </div>
      </div>

      <div class="grid cols-2">
        <div class="card">
          <div class="card-head">
            <h3>Top People</h3>
            <div class="spacer"></div>
            <button class="btn ghost sm" type="button" data-action="switch-tab" data-tab="people">View All (${fmtNum(st.persons || 0)})</button>
          </div>
          ${topPeopleHtml}
        </div>

        <div class="card">
          <div class="card-head">
            <h3>Smart Virtual Albums</h3>
            <div class="spacer"></div>
            <button class="btn ghost sm" type="button" data-action="switch-tab" data-tab="albums">Browse Albums</button>
          </div>
          <div style="display:grid;gap:9px">
            <div class="mini-person" data-action="open-album" data-album="group">
              <div class="mp-fallback" style="color:var(--green)">${icon('people')}</div>
              <div style="flex:1;min-width:0">
                <div style="font-weight:600;font-size:13.8px">Group Photos</div>
                <div class="muted" style="font-size:12.3px">2 or more distinct people together in one frame</div>
              </div>
              <span class="pill tabular">${fmtNum(albums.group || 0)}</span>
            </div>

            <div class="mini-person" data-action="open-album" data-album="unknown">
              <div class="mp-fallback" style="color:var(--warn)">${icon('unknown')}</div>
              <div style="flex:1;min-width:0">
                <div style="font-weight:600;font-size:13.8px">Unknown Faces</div>
                <div class="muted" style="font-size:12.3px">Ambiguous or single-shot faces ready for quick assignment</div>
              </div>
              <span class="pill warn tabular">${fmtNum(albums.unknown || 0)}</span>
            </div>

            <div class="mini-person" data-action="open-album" data-album="docs">
              <div class="mp-fallback" style="color:var(--info)">${icon('doc')}</div>
              <div style="flex:1;min-width:0">
                <div style="font-weight:600;font-size:13.8px">Text &amp; Documents</div>
                <div class="muted" style="font-size:12.3px">Receipts, notes, screenshots, and dense text captures</div>
              </div>
              <span class="pill info tabular">${fmtNum(albums.docs || 0)}</span>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function renderPipelineCardHtml(workers) {
    const running = state.pipeline.running;
    const actionType = (state.library && state.library.action) || 'start';
    const primaryText =
      actionType === 'resume' ? 'Resume Scan' : actionType === 'rescan' ? 'Re-scan Folder' : 'Start Scan';

    const curStage = running ? state.pipeline.stage_num || 1 : 0;
    const pct =
      running && state.pipeline.total > 0
        ? Math.min(100, Math.round((state.pipeline.done / state.pipeline.total) * 100))
        : 0;
    const indeterminate = running && (!state.pipeline.total || state.pipeline.total <= 0);

    return `
      <div class="pipeline-card ${running ? 'running' : ''}" id="pipelineCard">
        <div class="pipeline-top">
          <div>
            <div style="display:flex;align-items:center;gap:8px">
              <h3 id="pipeTitle">${
                running
                  ? esc(`${state.pipeline.stage_name || 'Running pipeline'} (Stage ${curStage} of 4)`)
                  : 'Indexing & Face Recognition Pipeline'
              }</h3>
              ${
                running
                  ? `<span class="pill ok"><span class="dot"></span>Active</span>`
                  : state.library && state.library.has_db
                  ? `<span class="pill">Ready</span>`
                  : `<span class="pill info">New Folder</span>`
              }
            </div>
            <div class="muted" style="font-size:13px;margin-top:2px" id="pipeSubtitle">
              ${
                running
                  ? esc(state.pipeline.detail || 'Working…')
                  : 'Scans images, extracts facial landmarks & 128-D embeddings, and clusters identities locally.'
              }
            </div>
          </div>

          <div class="spacer"></div>

          <div class="seg" role="group" aria-label="Speed dial">
            <button type="button" class="${state.speedMode === 'quiet' ? 'on' : ''}" data-action="set-speed" data-speed="quiet" ${running ? 'disabled' : ''} title="Quiet (${workers.quiet} worker)">Quiet (${workers.quiet})</button>
            <button type="button" class="${state.speedMode === 'normal' ? 'on' : ''}" data-action="set-speed" data-speed="normal" ${running ? 'disabled' : ''} title="Normal (${workers.normal} workers)">Normal (${workers.normal})</button>
            <button type="button" class="${state.speedMode === 'fast' ? 'on' : ''}" data-action="set-speed" data-speed="fast" ${running ? 'disabled' : ''} title="Fast (${workers.fast} workers)">Fast (${workers.fast})</button>
          </div>

          ${
            running
              ? `<button class="btn danger" type="button" data-action="pipeline-cancel" ${state.pipeline.cancelling ? 'disabled' : ''}>
                   ${icon('stop')}
                   <span>${state.pipeline.cancelling ? 'Stopping…' : 'Pause / Stop'}</span>
                 </button>`
              : `<button class="btn" type="button" data-action="pipeline-start">
                   ${icon(actionType === 'rescan' ? 'refresh' : 'play')}
                   <span>${esc(primaryText)}</span>
                 </button>`
          }
        </div>

        <div class="stage-steps" id="pipeStageSteps">
          ${STAGE_LABELS.map((s) => {
            let cls = '';
            if (running) {
              if (s.num < curStage) cls = 'done';
              else if (s.num === curStage) cls = 'active';
            } else if (state.library && state.library.has_db && state.library.action === 'rescan') {
              cls = 'done';
            }
            return `
              <div class="stage-step ${cls}" data-stage="${s.num}">
                <span class="s-num">${cls === 'done' ? '✓' : s.num}</span>
                <span class="truncate">${esc(s.short)}</span>
              </div>
            `;
          }).join('')}
        </div>

        ${
          running
            ? `
              <div class="progress-track">
                <div class="progress-fill ${indeterminate ? 'indeterminate' : ''}" id="pipeProgressFill" style="width:${pct}%"></div>
              </div>
              <div class="progress-meta tabular">
                <span id="pipeProgressText">${esc(state.pipeline.detail || 'Starting…')}</span>
                <span id="pipeProgressRight">${
                  state.pipeline.total > 0 ? `${fmtNum(state.pipeline.done)} / ${fmtNum(state.pipeline.total)} (${pct}%)` : ''
                }${state.pipeline.rss_mb ? ` · ${state.pipeline.rss_mb} MB RAM` : ''}</span>
              </div>
            `
            : ''
        }
      </div>
    `;
  }

  function updatePipelineDomInPlace() {
    // Update topbar scan status without clobbering main view
    const topScan = document.getElementById('topbarScanArea');
    if (topScan) {
      const p = state.pipeline;
      if (p.running) {
        const pct = p.total > 0 ? `${Math.min(100, Math.round((p.done / p.total) * 100))}%` : `Stage ${p.stage_num || 1}/4`;
        topScan.innerHTML = `
          <span class="pill ok tabular" title="${esc(p.detail)}">
            <span class="dot"></span>
            <span>${esc(p.stage_name || 'Scanning')} · ${esc(pct)}</span>
          </span>
          <button class="btn danger sm" type="button" data-action="pipeline-cancel" ${p.cancelling ? 'disabled' : ''}>
            ${icon('stop')}
            <span>${p.cancelling ? 'Stopping…' : 'Stop'}</span>
          </button>
        `;
      }
    }

    // Update Overview pipeline card in-place if mounted
    const card = document.getElementById('pipelineCard');
    if (!card || !state.pipeline.running) return;

    const p = state.pipeline;
    const pct = p.total > 0 ? Math.min(100, Math.round((p.done / p.total) * 100)) : 0;
    const indeterminate = !p.total || p.total <= 0;

    const titleEl = document.getElementById('pipeTitle');
    if (titleEl) titleEl.textContent = `${p.stage_name || 'Running pipeline'} (Stage ${p.stage_num || 1} of 4)`;

    const subEl = document.getElementById('pipeSubtitle');
    if (subEl) subEl.textContent = p.detail || 'Working…';

    const fillEl = document.getElementById('pipeProgressFill');
    if (fillEl) {
      fillEl.classList.toggle('indeterminate', indeterminate);
      if (!indeterminate) fillEl.style.width = `${pct}%`;
    }

    const txtEl = document.getElementById('pipeProgressText');
    if (txtEl) txtEl.textContent = p.detail || '';

    const rightEl = document.getElementById('pipeProgressRight');
    if (rightEl) {
      const countPart = p.total > 0 ? `${fmtNum(p.done)} / ${fmtNum(p.total)} (${pct}%)` : '';
      const rssPart = p.rss_mb ? `${countPart ? ' · ' : ''}${p.rss_mb} MB RAM` : '';
      rightEl.textContent = `${countPart}${rssPart}`;
    }

    const stepsEl = document.getElementById('pipeStageSteps');
    if (stepsEl) {
      for (const stepNode of stepsEl.querySelectorAll('.stage-step')) {
        const sNum = Number(stepNode.getAttribute('data-stage'));
        stepNode.classList.remove('done', 'active');
        const numSpan = stepNode.querySelector('.s-num');
        if (sNum < p.stage_num) {
          stepNode.classList.add('done');
          if (numSpan) numSpan.textContent = '✓';
        } else if (sNum === p.stage_num) {
          stepNode.classList.add('active');
          if (numSpan) numSpan.textContent = String(sNum);
        } else if (numSpan) {
          numSpan.textContent = String(sNum);
        }
      }
    }
  }

  // ---------------------------------------------------------------- 3. People Tab & Person Detail

  function getFilteredSortedPeople() {
    const q = state.peopleSearch.trim().toLowerCase();
    let list = state.people.filter((p) => {
      if (!q) return true;
      return String(p.display_name || '').toLowerCase().includes(q);
    });

    list = list.slice();
    switch (state.peopleSort) {
      case 'name_asc':
        list.sort((a, b) => String(a.display_name).localeCompare(String(b.display_name), undefined, { numeric: true }));
        break;
      case 'named_first':
        list.sort(
          (a, b) =>
            Number(b.name_is_user_set) - Number(a.name_is_user_set) ||
            b.photo_count - a.photo_count ||
            a.id - b.id
        );
        break;
      case 'unnamed_first':
        list.sort(
          (a, b) =>
            Number(a.name_is_user_set) - Number(b.name_is_user_set) ||
            b.photo_count - a.photo_count ||
            a.id - b.id
        );
        break;
      case 'most_photos':
      default:
        list.sort((a, b) => b.photo_count - a.photo_count || b.face_count - a.face_count || a.id - b.id);
        break;
    }
    return list;
  }

  function renderPeopleGridInnerHtml() {
    const filtered = getFilteredSortedPeople();
    if (!state.people.length) {
      return `
        <div class="card">
          <div class="empty">
            <strong>No people discovered yet</strong>
            Run the indexing pipeline on your photo folder to detect faces and automatically cluster people.
            <div style="margin-top:14px">
              <button class="btn" type="button" data-action="pipeline-start">${icon('play')}<span>Start Scan</span></button>
            </div>
          </div>
        </div>
      `;
    }

    if (!filtered.length) {
      return `
        <div class="card">
          <div class="empty">
            <strong>No matching people</strong>
            No person matches "${esc(state.peopleSearch)}".
          </div>
        </div>
      `;
    }

    return `
      <div class="people-grid">
        ${filtered
          .map((p) => {
            const isRenaming = state.inlineRenameId === p.id;
            return `
              <div class="person-card" data-action="open-person" data-person-id="${p.id}">
                ${
                  p.name_is_user_set
                    ? `<span class="person-named-dot pill ok" style="padding:1px 7px;font-size:10.5px">Named</span>`
                    : ''
                }
                <div class="person-avatar">
                  ${
                    p.cover_thumb_path
                      ? `<img src="${esc(mediaSrc(p.cover_thumb_path))}" alt="${esc(p.display_name)}" loading="lazy" />`
                      : icon('person')
                  }
                </div>

                ${
                  isRenaming
                    ? `<input
                         type="text"
                         class="inline-rename-input"
                         id="inlineRenameInput"
                         data-person-id="${p.id}"
                         value="${esc(p.display_name)}"
                         maxlength="80"
                       />`
                    : `<div class="person-name" title="${esc(p.display_name)}">${esc(p.display_name)}</div>`
                }

                <div class="person-count">${fmtNum(p.photo_count)} photo${p.photo_count === 1 ? '' : 's'} · ${fmtNum(p.face_count)} face${p.face_count === 1 ? '' : 's'}</div>

                <div class="person-actions">
                  <button class="icon-btn" type="button" data-action="start-inline-rename" data-person-id="${p.id}" title="Rename person">
                    ${icon('edit')}
                  </button>
                  <button class="icon-btn" type="button" data-action="open-merge-modal" data-person-id="${p.id}" title="Merge into another person…">
                    ${icon('merge')}
                  </button>
                  <button class="icon-btn danger" type="button" data-action="confirm-delete-person" data-person-id="${p.id}" title="Delete person cluster">
                    ${icon('trash')}
                  </button>
                </div>
              </div>
            `;
          })
          .join('')}
      </div>
    `;
  }

  function renderPeopleView() {
    const namedCount = state.people.filter((p) => p.name_is_user_set).length;
    return `
      <div class="page-head">
        <div>
          <h1>People</h1>
          <p>${fmtNum(state.people.length)} people clusters · ${fmtNum(namedCount)} named by you</p>
        </div>
        <div class="spacer"></div>
        <div class="btn-row">
          <button class="btn ghost sm" type="button" data-action="recluster-auto-faces" ${state.pipeline.running || !state.people.length ? 'disabled' : ''} title="Re-run graph clustering on auto-assigned faces while preserving manual assignments">
            ${icon('refresh')}
            <span>Re-cluster Auto Faces</span>
          </button>
          <button class="btn soft sm" type="button" data-action="open-export-modal" ${!state.people.length ? 'disabled' : ''}>
            ${icon('export')}
            <span>Export People…</span>
          </button>
        </div>
      </div>

      <div class="filters-bar">
        <div class="search-box" style="flex:1;max-width:340px">
          ${icon('search')}
          <input
            type="search"
            id="peopleSearchInput"
            placeholder="Search people by name…"
            value="${esc(state.peopleSearch)}"
          />
        </div>

        <div style="width:195px">
          <select id="peopleSortSelect" aria-label="Sort people">
            <option value="most_photos" ${state.peopleSort === 'most_photos' ? 'selected' : ''}>Most photos first</option>
            <option value="named_first" ${state.peopleSort === 'named_first' ? 'selected' : ''}>Named people first</option>
            <option value="unnamed_first" ${state.peopleSort === 'unnamed_first' ? 'selected' : ''}>Unnamed clusters first</option>
            <option value="name_asc" ${state.peopleSort === 'name_asc' ? 'selected' : ''}>Name (A–Z)</option>
          </select>
        </div>
      </div>

      <div id="peopleGridHost">
        ${renderPeopleGridInnerHtml()}
      </div>
    `;
  }

  function focusInlineRenameIfAny() {
    if (!state.inlineRenameId) return;
    const inp = document.getElementById('inlineRenameInput');
    if (inp) {
      inp.focus();
      inp.select();
    }
  }

  function renderPersonDetailView() {
    const { person, faces } = state.personDetail;
    const selCount = state.selectedFaceIds.size;

    const selectionBarHtml =
      selCount > 0
        ? `
          <div class="selection-bar">
            <strong>${fmtNum(selCount)} face${selCount === 1 ? '' : 's'} selected</strong>
            <div class="spacer"></div>
            <div class="btn-row">
              <button class="btn sm" type="button" data-action="open-reassign-faces-modal">
                ${icon('people')}
                <span>Move to Person…</span>
              </button>
              <button class="btn ghost sm" type="button" data-action="open-new-person-modal">
                ${icon('plus')}
                <span>New Person…</span>
              </button>
              <button class="btn danger sm" type="button" data-action="batch-unassign-faces" title="Remove from this person and move to Unknown Faces">
                ${icon('unlink')}
                <span>Not ${esc(person.display_name)}</span>
              </button>
              <button class="btn ghost sm" type="button" data-action="clear-face-selection">Clear</button>
            </div>
          </div>
        `
        : '';

    return `
      <div style="margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;gap:10px">
        <button class="btn ghost sm" type="button" data-action="close-person-detail">
          ${icon('arrowLeft')}
          <span>All People</span>
        </button>
        <div class="btn-row">
          <button class="btn ghost sm" type="button" data-action="select-all-faces">
            ${selCount === faces.length && faces.length > 0 ? 'Deselect All' : `Select All (${fmtNum(faces.length)})`}
          </button>
          <button class="btn ghost sm" type="button" data-action="filter-photos-by-person" data-person-id="${person.id}">
            ${icon('photo')}
            <span>View Full Photos</span>
          </button>
        </div>
      </div>

      <div class="detail-banner">
        <div class="person-avatar">
          ${
            person.cover_thumb_path
              ? `<img src="${esc(mediaSrc(person.cover_thumb_path))}" alt="${esc(person.display_name)}" />`
              : icon('person')
          }
        </div>
        <div style="min-width:180px;flex:1">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <h2>${esc(person.display_name)}</h2>
            ${
              person.name_is_user_set
                ? `<span class="pill ok">Verified name</span>`
                : `<span class="pill">Auto-labeled</span>`
            }
          </div>
          <div class="muted tabular" style="font-size:13.3px;margin-top:3px">
            Appears in ${fmtNum(person.photo_count)} photo${person.photo_count === 1 ? '' : 's'} · ${fmtNum(faces.length)} detected face${faces.length === 1 ? '' : 's'}
          </div>
        </div>

        <div class="btn-row">
          <button class="btn ghost sm" type="button" data-action="open-rename-modal" data-person-id="${person.id}">
            ${icon('edit')}
            <span>Rename</span>
          </button>
          <button class="btn ghost sm" type="button" data-action="open-merge-modal" data-person-id="${person.id}">
            ${icon('merge')}
            <span>Merge Into…</span>
          </button>
          <button class="btn ghost sm" type="button" data-action="recheck-person-album" data-person-id="${person.id}" title="Active learning: re-evaluate auto-assigned faces against your manual corrections">
            ${icon('sparkle')}
            <span>Re-check Album</span>
          </button>
          <button class="btn soft sm" type="button" data-action="open-export-person" data-person-id="${person.id}">
            ${icon('export')}
            <span>Export</span>
          </button>
          <button class="btn danger sm" type="button" data-action="confirm-delete-person" data-person-id="${person.id}">
            ${icon('trash')}
            <span>Delete</span>
          </button>
        </div>
      </div>

      ${selectionBarHtml}

      <div class="faces-grid ${selCount > 0 ? 'selection-active' : ''}">
        ${faces
          .map((f, idx) => {
            const isSel = state.selectedFaceIds.has(f.id);
            const thumb = f.thumb_path || f.image_thumb || f.image_path;
            return `
              <div class="face-tile ${isSel ? 'selected' : ''}" data-face-id="${f.id}">
                <div class="face-tile-img" data-action="open-person-face-lightbox" data-idx="${idx}">
                  <button
                    class="face-select-check"
                    type="button"
                    data-action="toggle-face-select"
                    data-face-id="${f.id}"
                    data-idx="${idx}"
                    title="Select face (Shift+Click for range)"
                  >
                    ${isSel ? icon('check') : ''}
                  </button>
                  ${f.is_cover ? `<span class="face-cover-badge">Cover</span>` : ''}
                  <img src="${esc(mediaSrc(thumb))}" alt="${esc(f.filename)}" loading="lazy" />
                </div>
                <div class="face-tile-foot">
                  <span class="truncate" title="${esc(f.filename)}">${
                    f.assign_source === 'manual' ? 'Manual' : `${Math.round((f.det_score || 0) * 100)}%`
                  }</span>
                  ${
                    f.is_cover
                      ? `<span class="muted" style="font-size:11px">★ Cover</span>`
                      : `<button type="button" data-action="set-person-cover" data-person-id="${person.id}" data-face-id="${f.id}" title="Set as person cover avatar">Set cover</button>`
                  }
                </div>
              </div>
            `;
          })
          .join('')}
      </div>
    `;
  }

  // ---------------------------------------------------------------- 4. Virtualized Photos Tab

  function renderPhotosView() {
    const personOptions = state.people
      .map(
        (p) =>
          `<option value="${p.id}" ${String(state.photosPersonId) === String(p.id) ? 'selected' : ''}>${esc(p.display_name)} (${fmtNum(p.photo_count)})</option>`
      )
      .join('');

    return `
      <div class="page-head" style="margin-bottom:12px">
        <div>
          <h1>All Photos</h1>
          <p>Virtualized gallery — click any photo to inspect full resolution, face boxes, and similarity matches</p>
        </div>
        <div class="spacer"></div>
        <span class="pill tabular" id="photosCountPill">${fmtNum(state.photos.length)} shown</span>
      </div>

      <div class="filters-bar">
        <div class="seg" role="group" aria-label="Photo filter">
          <button type="button" class="${state.photosFilter === 'all' ? 'on' : ''}" data-action="set-photos-filter" data-filter="all">All</button>
          <button type="button" class="${state.photosFilter === 'with_faces' ? 'on' : ''}" data-action="set-photos-filter" data-filter="with_faces">With Faces</button>
          <button type="button" class="${state.photosFilter === 'no_faces' ? 'on' : ''}" data-action="set-photos-filter" data-filter="no_faces">No Faces</button>
          <button type="button" class="${state.photosFilter === 'docs' ? 'on' : ''}" data-action="set-photos-filter" data-filter="docs">Docs</button>
          <button type="button" class="${state.photosFilter === 'errors' ? 'on' : ''}" data-action="set-photos-filter" data-filter="errors">Errors</button>
        </div>

        <div style="width:205px">
          <select id="photosPersonSelect" aria-label="Filter by person">
            <option value="">All people</option>
            ${personOptions}
          </select>
        </div>

        <div class="search-box" style="flex:1;min-width:190px;max-width:300px">
          ${icon('search')}
          <input
            type="search"
            id="photosSearchInput"
            placeholder="Filter by filename or path…"
            value="${esc(state.photosSearch)}"
          />
        </div>

        <div class="spacer"></div>

        <div class="seg" role="group" aria-label="Thumbnail size">
          <button type="button" class="${state.photoGridSize === 'compact' ? 'on' : ''}" data-action="set-grid-size" data-size="compact">S</button>
          <button type="button" class="${state.photoGridSize === 'comfortable' ? 'on' : ''}" data-action="set-grid-size" data-size="comfortable">M</button>
          <button type="button" class="${state.photoGridSize === 'large' ? 'on' : ''}" data-action="set-grid-size" data-size="large">L</button>
        </div>
      </div>

      <div class="vgrid-viewport" id="vgridViewport">
        <div class="vgrid-spacer" id="vgridSpacer"></div>
      </div>
    `;
  }

  function mountVirtualPhotosGrid() {
    destroyVGrid();
    const viewport = document.getElementById('vgridViewport');
    const spacer = document.getElementById('vgridSpacer');
    if (!viewport || !spacer) return;

    const renderedNodes = new Map();
    let rafPending = false;

    function layout() {
      rafPending = false;
      const items = state.photos;
      const countPill = document.getElementById('photosCountPill');
      if (countPill) countPill.textContent = `${fmtNum(items.length)} shown`;

      if (!items.length) {
        for (const node of renderedNodes.values()) node.remove();
        renderedNodes.clear();
        spacer.style.height = 'auto';
        spacer.innerHTML = `
          <div class="empty">
            <strong>${state.photosLoading ? 'Loading photos…' : 'No photos match current filter'}</strong>
            ${
              state.photosLoading
                ? 'Reading SQLite index…'
                : state.library && !state.library.has_db
                ? 'Start a scan from the Overview tab to index thumbnails and faces.'
                : 'Try clearing search filters or selecting another category.'
            }
          </div>
        `;
        return;
      }

      // Clear any empty-state placeholder if items exist
      const emptyEl = spacer.querySelector('.empty');
      if (emptyEl) emptyEl.remove();

      const gap = 10;
      const target = GRID_SIZES[state.photoGridSize] || 158;
      const innerWidth = Math.max(200, viewport.clientWidth - 24);
      const cols = Math.max(1, Math.floor((innerWidth + gap) / (target + gap)));
      const cellSize = Math.max(84, Math.floor((innerWidth - gap * (cols - 1)) / cols));
      const rowPitch = cellSize + gap;
      const totalRows = Math.ceil(items.length / cols);
      const totalHeight = Math.max(0, totalRows * rowPitch - gap);

      spacer.style.height = `${totalHeight}px`;

      const scrollTop = viewport.scrollTop;
      const viewH = viewport.clientHeight || 600;
      const firstRow = Math.max(0, Math.floor(scrollTop / rowPitch) - 2);
      const lastRow = Math.min(totalRows - 1, Math.ceil((scrollTop + viewH) / rowPitch) + 2);

      const startIdx = firstRow * cols;
      const endIdx = Math.min(items.length - 1, (lastRow + 1) * cols - 1);

      const neededKeys = new Set();
      for (let i = startIdx; i <= endIdx; i++) {
        const photo = items[i];
        const key = String(photo.id);
        neededKeys.add(key);

        const row = Math.floor(i / cols);
        const col = i % cols;
        const left = col * (cellSize + gap);
        const top = row * rowPitch;

        let cell = renderedNodes.get(key);
        if (!cell) {
          cell = document.createElement('div');
          cell.className = 'photo-cell';
          cell.dataset.action = 'open-grid-photo';
          cell.title = photo.filename || '';

          if (photo.status === 'error') {
            cell.innerHTML = `<div class="pc-err"><div><strong>Unreadable</strong><br/>${esc(photo.filename)}</div></div>`;
          } else {
            const src = mediaSrc(photo.thumb_path || photo.path);
            const badge =
              photo.face_count > 0
                ? `<span class="pc-badge">${photo.face_count} face${photo.face_count === 1 ? '' : 's'}</span>`
                : photo.doc_score >= 0.5
                ? `<span class="pc-badge">Doc</span>`
                : '';
            cell.innerHTML = `<img src="${esc(src)}" alt="${esc(photo.filename)}" loading="lazy" />${badge}`;
          }
          spacer.appendChild(cell);
          renderedNodes.set(key, cell);
        }

        cell.dataset.idx = String(i);
        cell.style.width = `${cellSize}px`;
        cell.style.height = `${cellSize}px`;
        cell.style.transform = `translate(${left}px, ${top}px)`;
      }

      for (const [key, node] of renderedNodes.entries()) {
        if (!neededKeys.has(key)) {
          node.remove();
          renderedNodes.delete(key);
        }
      }
    }

    function scheduleLayout() {
      if (rafPending) return;
      rafPending = true;
      requestAnimationFrame(layout);
    }

    viewport.addEventListener('scroll', scheduleLayout, { passive: true });
    const ro = new ResizeObserver(scheduleLayout);
    ro.observe(viewport);

    layout();

    vgridCleanup = () => {
      viewport.removeEventListener('scroll', scheduleLayout);
      ro.disconnect();
      renderedNodes.clear();
    };
  }

  // ---------------------------------------------------------------- 5. Smart Albums & Export Tab

  function renderAlbumsView() {
    const albumMeta = {
      group: { icon: 'people', pillCls: 'ok' },
      unknown: { icon: 'unknown', pillCls: 'warn' },
      docs: { icon: 'doc', pillCls: 'info' }
    };

    const cardsHtml = (state.albumsSummary || [])
      .map((a) => {
        const meta = albumMeta[a.id] || { icon: 'albums', pillCls: '' };
        const previews = (a.previews || []).slice(0, 4);
        let mosaicHtml = '';
        if (!previews.length) {
          mosaicHtml = `<div class="album-mosaic"><div class="album-mosaic-empty">${icon(meta.icon)}</div></div>`;
        } else if (previews.length === 1) {
          mosaicHtml = `<div class="album-mosaic single"><img src="${esc(mediaSrc(previews[0]))}" alt="${esc(a.title)}" loading="lazy" /></div>`;
        } else {
          mosaicHtml = `
            <div class="album-mosaic">
              ${previews.map((p) => `<img src="${esc(mediaSrc(p))}" alt="" loading="lazy" />`).join('')}
            </div>
          `;
        }

        return `
          <div class="album-card" data-action="open-album" data-album="${esc(a.id)}">
            ${mosaicHtml}
            <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
              <h3>${esc(a.title)}</h3>
              <span class="pill ${meta.pillCls} tabular">${fmtNum(a.count)}</span>
            </div>
            <p class="muted" style="font-size:13px;margin:0">${esc(a.subtitle || '')}</p>
          </div>
        `;
      })
      .join('');

    const exp = state.exportJob;
    const exportStatusHtml = exp.running
      ? `
        <div class="note ok" id="exportLiveBanner">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap">
            <div>
              <strong>Exporting clean copies…</strong>
              <span id="exportLiveText">${esc(exp.detail || 'Copying files…')}</span>
            </div>
            <button class="btn danger sm" type="button" data-action="export-cancel" ${exp.cancelling ? 'disabled' : ''}>
              ${icon('stop')}<span>${exp.cancelling ? 'Stopping…' : 'Cancel Export'}</span>
            </button>
          </div>
          <div class="progress-track" style="margin-top:10px">
            <div class="progress-fill" id="exportLiveFill" style="width:${exp.total > 0 ? Math.round((exp.done / exp.total) * 100) : 0}%"></div>
          </div>
        </div>
      `
      : exp.lastResult
      ? `
        <div class="note ${exp.lastResult.cancelled ? 'warn' : 'ok'}">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap">
            <div>
              <strong>${exp.lastResult.cancelled ? 'Export cancelled.' : 'Export complete!'}</strong>
              Copied ${fmtNum(exp.lastResult.copied)} photo${exp.lastResult.copied === 1 ? '' : 's'} into organized folders (${fmtNum(exp.lastResult.skipped)} already up-to-date).
            </div>
            ${
              exp.lastResult.export_root
                ? `<button class="btn ghost sm" type="button" data-action="open-in-explorer" data-path="${esc(exp.lastResult.export_root)}">
                     ${icon('external')}<span>Open Export Folder</span>
                   </button>`
                : ''
            }
          </div>
        </div>
      `
      : '';

    return `
      <div class="page-head">
        <div>
          <h1>Smart Albums &amp; Safe Export</h1>
          <p>Automatic virtual collections and copy-only folder export</p>
        </div>
        <div class="spacer"></div>
        <button class="btn" type="button" data-action="open-export-modal" ${!(state.library && state.library.has_db) ? 'disabled' : ''}>
          ${icon('export')}
          <span>Export Sorted Copies…</span>
        </button>
      </div>

      ${exportStatusHtml}

      <div class="grid cols-3" style="margin-bottom:20px">
        ${cardsHtml}
      </div>

      <div class="card">
        <div class="card-head">
          ${icon('shield')}
          <h3>Copy-Only Export Architecture</h3>
        </div>
        <p class="muted" style="font-size:13.6px;max-width:78ch">
          When you export, Auto-Sorter creates clean subfolders for each person (<code class="mono">Alice/</code>, <code class="mono">Bob/</code>) plus optional <code class="mono">Group Photos/</code>, <code class="mono">Text and Documents/</code>, and <code class="mono">Unknown Faces/</code> folders in your chosen destination. Multi-person photos are copied into each relevant folder safely while your source library remains untouched.
        </p>
        <div class="btn-row" style="margin-top:12px">
          <button class="btn soft" type="button" data-action="open-export-modal" ${!(state.library && state.library.has_db) ? 'disabled' : ''}>
            ${icon('export')}
            <span>Configure &amp; Start Export…</span>
          </button>
        </div>
      </div>
    `;
  }

  function renderAlbumDetailView() {
    const detail = state.albumDetail;
    const titles = {
      group: 'Group Photos',
      unknown: 'Unknown Faces',
      docs: 'Text & Documents'
    };
    const subtitles = {
      group: 'Photos containing 2 or more distinct identified people',
      unknown: 'Click faces to inspect or select multiple faces to assign to an existing or new person',
      docs: 'Images scored as documents, receipts, screenshots, or dense text'
    };

    const items = detail.items || [];
    const isUnknown = detail.album_type === 'unknown';
    const selCount = state.selectedFaceIds.size;

    const unknownBarHtml =
      isUnknown && selCount > 0
        ? `
          <div class="selection-bar">
            <strong>${fmtNum(selCount)} unknown face${selCount === 1 ? '' : 's'} selected</strong>
            <div class="spacer"></div>
            <div class="btn-row">
              <button class="btn sm" type="button" data-action="open-reassign-faces-modal">
                ${icon('people')}
                <span>Assign to Person…</span>
              </button>
              <button class="btn ghost sm" type="button" data-action="open-new-person-modal">
                ${icon('plus')}
                <span>Create New Person…</span>
              </button>
              <button class="btn ghost sm" type="button" data-action="clear-face-selection">Clear</button>
            </div>
          </div>
        `
        : '';

    return `
      <div style="margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;gap:10px">
        <button class="btn ghost sm" type="button" data-action="close-album-detail">
          ${icon('arrowLeft')}
          <span>All Smart Albums</span>
        </button>
        ${
          isUnknown && items.length > 0
            ? `<button class="btn ghost sm" type="button" data-action="select-all-unknown-faces">
                 ${selCount === items.length ? 'Deselect All' : `Select All (${fmtNum(items.length)})`}
               </button>`
            : ''
        }
      </div>

      <div class="page-head">
        <div>
          <h1>${esc(titles[detail.album_type] || 'Album')}</h1>
          <p>${esc(subtitles[detail.album_type] || '')} · ${fmtNum(items.length)} item${items.length === 1 ? '' : 's'}</p>
        </div>
      </div>

      ${unknownBarHtml}

      ${
        !items.length
          ? `<div class="card"><div class="empty"><strong>This smart album is empty</strong>No items currently match this album's criteria.</div></div>`
          : isUnknown
          ? `
            <div class="faces-grid ${selCount > 0 ? 'selection-active' : ''}">
              ${items
                .map((it, idx) => {
                  const isSel = state.selectedFaceIds.has(it.face_id);
                  const src = mediaSrc(it.thumb_path || it.image_thumb || it.path);
                  return `
                    <div class="face-tile ${isSel ? 'selected' : ''}">
                      <div class="face-tile-img" data-action="open-album-lightbox" data-idx="${idx}">
                        <button
                          class="face-select-check"
                          type="button"
                          data-action="toggle-unknown-face-select"
                          data-face-id="${it.face_id}"
                          data-idx="${idx}"
                          title="Select unknown face"
                        >
                          ${isSel ? icon('check') : ''}
                        </button>
                        <img src="${esc(src)}" alt="${esc(it.filename)}" loading="lazy" />
                      </div>
                      <div class="face-tile-foot">
                        <span class="truncate" title="${esc(it.filename)}">${esc(it.filename)}</span>
                        <button type="button" data-action="quick-assign-single-face" data-face-id="${it.face_id}">Assign…</button>
                      </div>
                    </div>
                  `;
                })
                .join('')}
            </div>
          `
          : `
            <div class="faces-grid" style="grid-template-columns:repeat(auto-fill, minmax(176px, 1fr))">
              ${items
                .map(
                  (it, idx) => `
                  <div class="face-tile" data-action="open-album-lightbox" data-idx="${idx}" style="cursor:pointer">
                    <div class="face-tile-img">
                      <img src="${esc(mediaSrc(it.thumb_path || it.path))}" alt="${esc(it.filename)}" loading="lazy" />
                    </div>
                    <div class="face-tile-foot">
                      <span class="truncate" title="${esc(it.filename)}">${esc(it.filename)}</span>
                      ${it.doc_score !== undefined ? `<span class="mono">${Math.round(it.doc_score * 100)}%</span>` : ''}
                    </div>
                  </div>
                `
                )
                .join('')}
            </div>
          `
      }
    `;
  }

  // ---------------------------------------------------------------- 6. Settings & Diagnostics Tab

  function renderSettingsView() {
    const sys = state.sysInfo || {};
    const workers = sys.speed_workers || { quiet: 1, normal: 2, fast: 4 };
    const yunet = (sys.models && sys.models.yunet) || {};
    const sface = (sys.models && sys.models.sface) || {};

    return `
      <div class="page-head">
        <div>
          <h1>Settings &amp; Diagnostics</h1>
          <p>Library storage paths, worker concurrency, and local ONNX model verification</p>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3>Active Library Paths</h3>
          <div class="spacer"></div>
          <button class="btn ghost sm" type="button" data-action="open-switch-library">
            ${icon('folder')}<span>Switch Photo Library…</span>
          </button>
        </div>

        <div class="field">
          <label>Source Photo Folder (Read-only — never modified)</label>
          <div class="copybox">
            <code>${esc(state.photoFolder)}</code>
            <button class="btn ghost sm" type="button" data-action="copy-text" data-text="${esc(state.photoFolder)}" data-label="Photo folder path">${icon('copy')}<span>Copy</span></button>
            <button class="btn ghost sm" type="button" data-action="open-in-explorer" data-path="${esc(state.photoFolder)}">${icon('external')}<span>Open</span></button>
          </div>
        </div>

        <div class="field" style="margin-bottom:0">
          <label>Index, Thumbnail Cache &amp; Logs Folder (<code class="mono">.autosorter</code>)</label>
          <div class="copybox">
            <code>${esc(state.dataFolder)}</code>
            <button class="btn ghost sm" type="button" data-action="copy-text" data-text="${esc(state.dataFolder)}" data-label="Data folder path">${icon('copy')}<span>Copy</span></button>
            <button class="btn ghost sm" type="button" data-action="open-in-explorer" data-path="${esc(state.dataFolder)}">${icon('external')}<span>Open</span></button>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3>Performance &amp; Clustering</h3>
        </div>

        <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;padding-bottom:14px;border-bottom:1px solid var(--line-soft)">
          <div>
            <strong>Pipeline Speed Mode</strong>
            <div class="muted" style="font-size:13px">Controls background worker thread concurrency during face detection &amp; embedding.</div>
          </div>
          <div class="seg">
            <button type="button" class="${state.speedMode === 'quiet' ? 'on' : ''}" data-action="set-speed" data-speed="quiet" ${state.pipeline.running ? 'disabled' : ''}>Quiet (${workers.quiet})</button>
            <button type="button" class="${state.speedMode === 'normal' ? 'on' : ''}" data-action="set-speed" data-speed="normal" ${state.pipeline.running ? 'disabled' : ''}>Normal (${workers.normal})</button>
            <button type="button" class="${state.speedMode === 'fast' ? 'on' : ''}" data-action="set-speed" data-speed="fast" ${state.pipeline.running ? 'disabled' : ''}>Fast (${workers.fast})</button>
          </div>
        </div>

        <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;padding-top:14px">
          <div>
            <strong>Rebuild Auto Face Clusters</strong>
            <div class="muted" style="font-size:13px">Re-runs mutual-kNN connected-components clustering across auto-assigned faces while preserving manual assignments &amp; names.</div>
          </div>
          <button class="btn ghost" type="button" data-action="recluster-auto-faces" ${state.pipeline.running || !(state.library && state.library.has_db) ? 'disabled' : ''}>
            ${icon('refresh')}
            <span>Re-cluster Auto Faces</span>
          </button>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3>Local Engine &amp; Bundled ONNX Models</h3>
          <div class="spacer"></div>
          <span class="pill ${state.bridgeConnected ? 'ok' : 'bad'}">
            <span class="dot"></span>${state.bridgeConnected ? 'Python Engine Connected' : 'Reconnecting…'}
          </span>
        </div>

        <dl class="kv">
          <dt>App Version</dt>
          <dd class="mono">${esc(sys.app_version || '—')}</dd>

          <dt>Python Runtime</dt>
          <dd class="mono">Python ${esc(sys.python_version || '—')} (${esc(sys.platform || '')})</dd>

          <dt>Logical CPUs</dt>
          <dd class="tabular">${fmtNum(sys.cpu_count)} cores (Quiet: ${workers.quiet}, Normal: ${workers.normal}, Fast: ${workers.fast})</dd>

          <dt>Engine Memory (RSS)</dt>
          <dd class="tabular">${sys.rss_mb ? `${sys.rss_mb} MB` : '—'}</dd>

          <dt>Face Detector</dt>
          <dd>
            <span class="pill ${yunet.present ? 'ok' : 'bad'}">${yunet.present ? 'Loaded' : 'Missing'}</span>
            <span class="mono muted" style="margin-left:8px">${esc(yunet.name || 'face_detection_yunet_2023mar.onnx')} (${yunet.size_kb || 0} KB)</span>
          </dd>

          <dt>Face Recognizer</dt>
          <dd>
            <span class="pill ${sface.present ? 'ok' : 'bad'}">${sface.present ? 'Loaded' : 'Missing'}</span>
            <span class="mono muted" style="margin-left:8px">${esc(sface.name || 'face_recognition_sface_2021dec.onnx')} (${sface.size_mb || 0} MB)</span>
          </dd>
        </dl>
      </div>
    `;
  }

  // ---------------------------------------------------------------- Modals

  function closeModal() {
    state.modal = null;
    dom.modalHost.innerHTML = '';
  }

  function openSwitchLibraryModal() {
    state.modal = {
      type: 'switch-library',
      photoFolder: state.photoFolder || '',
      customDataFolder: state.customDataFolder || ''
    };
    renderModal();
  }

  function openRenameModal(personId) {
    const p =
      state.people.find((x) => x.id === personId) ||
      (state.personDetail && state.personDetail.person.id === personId ? state.personDetail.person : null);
    if (!p) return;
    state.modal = {
      type: 'rename-person',
      personId: p.id,
      currentName: p.display_name
    };
    renderModal();
  }

  function openMergeModal(sourcePersonId) {
    const src =
      state.people.find((x) => x.id === sourcePersonId) ||
      (state.personDetail && state.personDetail.person.id === sourcePersonId ? state.personDetail.person : null);
    if (!src) return;
    const candidates = state.people.filter((p) => p.id !== sourcePersonId);
    state.modal = {
      type: 'merge-person',
      sourcePerson: src,
      targetPersonId: candidates.length ? candidates[0].id : null,
      search: ''
    };
    renderModal();
  }

  function openDeletePersonModal(personId) {
    const p =
      state.people.find((x) => x.id === personId) ||
      (state.personDetail && state.personDetail.person.id === personId ? state.personDetail.person : null);
    if (!p) return;
    state.modal = {
      type: 'delete-person',
      person: p
    };
    renderModal();
  }

  function openReassignFacesModal(explicitFaceIds = null) {
    const faceIds = explicitFaceIds || Array.from(state.selectedFaceIds);
    if (!faceIds.length) return;
    const excludeId =
      state.tab === 'people' && state.selectedPersonId && !state.lightbox.open ? state.selectedPersonId : null;
    const candidates = state.people.filter((p) => p.id !== excludeId);
    state.modal = {
      type: 'reassign-faces',
      faceIds,
      excludePersonId: excludeId,
      targetPersonId: candidates.length ? candidates[0].id : null,
      search: ''
    };
    renderModal();
  }

  function openNewPersonModal(explicitFaceIds = null) {
    const faceIds = explicitFaceIds || Array.from(state.selectedFaceIds);
    if (!faceIds.length) return;
    state.modal = {
      type: 'new-person',
      faceIds,
      displayName: ''
    };
    renderModal();
  }

  function openExportModal(preselectedPersonIds = null) {
    const hasSubset = Array.isArray(preselectedPersonIds) && preselectedPersonIds.length > 0;
    state.modal = {
      type: 'export',
      exportRoot: state.lastExportFolder || '',
      scope: hasSubset ? 'selected' : 'all',
      selectedPersonIds: new Set(hasSubset ? preselectedPersonIds : state.people.map((p) => p.id)),
      includeGroup: !hasSubset,
      includeDocs: !hasSubset,
      includeUnknown: false,
      search: ''
    };
    renderModal();
  }

  function renderModal() {
    const m = state.modal;
    if (!m) {
      dom.modalHost.innerHTML = '';
      return;
    }

    let inner = '';

    if (m.type === 'switch-library') {
      const recents = state.recentLibraries
        .map(
          (r, idx) => `
          <div class="picker-row" data-action="modal-open-recent" data-idx="${idx}">
            <div class="pr-avatar" style="color:var(--green)">${icon('folderOpen')}</div>
            <div style="flex:1;min-width:0">
              <div class="truncate" style="font-weight:600;font-size:13.5px">${esc(baseName(r.photoFolder))}</div>
              <div class="mono muted truncate" style="font-size:11.5px">${esc(r.photoFolder)}</div>
            </div>
          </div>
        `
        )
        .join('');

      inner = `
        <div class="modal wide">
          <div class="modal-head">
            <h3>Open / Switch Photo Library</h3>
            <div class="spacer"></div>
            <button class="icon-btn" type="button" data-action="close-modal">${icon('close')}</button>
          </div>
          <div class="modal-body">
            <div class="field">
              <label>Photo Folder</label>
              <div class="btn-row" style="flex-wrap:nowrap">
                <input type="text" id="modalPhotoFolderInput" value="${esc(m.photoFolder)}" placeholder="Select a folder of photos…" />
                <button class="btn ghost" type="button" data-action="modal-browse-photo">Browse…</button>
              </div>
            </div>
            <div class="field">
              <label>Optional Custom Index Folder (<code class="mono">.autosorter</code>)</label>
              <div class="btn-row" style="flex-wrap:nowrap">
                <input type="text" id="modalCustomDataInput" value="${esc(m.customDataFolder)}" placeholder="Leave blank to use <photo-folder>\\.autosorter" />
                <button class="btn ghost" type="button" data-action="modal-browse-data">Browse…</button>
              </div>
            </div>
            ${
              recents
                ? `<div class="section-title">Recent Libraries</div><div class="picker-list">${recents}</div>`
                : ''
            }
          </div>
          <div class="modal-foot">
            <button class="btn ghost" type="button" data-action="close-modal">Cancel</button>
            <button class="btn" type="button" data-action="submit-switch-library">Open Library</button>
          </div>
        </div>
      `;
    } else if (m.type === 'rename-person') {
      inner = `
        <div class="modal">
          <div class="modal-head">
            <h3>Rename Person</h3>
            <div class="spacer"></div>
            <button class="icon-btn" type="button" data-action="close-modal">${icon('close')}</button>
          </div>
          <form id="modalRenameForm">
            <div class="modal-body">
              <div class="field" style="margin-bottom:0">
                <label>Display Name</label>
                <input type="text" id="modalRenameInput" value="${esc(m.currentName)}" maxlength="80" required />
                <div class="hint">Used as the folder name when exporting photos of this person.</div>
              </div>
            </div>
            <div class="modal-foot">
              <button class="btn ghost" type="button" data-action="close-modal">Cancel</button>
              <button class="btn" type="submit">Save Name</button>
            </div>
          </form>
        </div>
      `;
    } else if (m.type === 'merge-person') {
      const candidates = state.people.filter(
        (p) =>
          p.id !== m.sourcePerson.id &&
          (!m.search || String(p.display_name).toLowerCase().includes(m.search.toLowerCase()))
      );
      inner = `
        <div class="modal">
          <div class="modal-head">
            <h3>Merge "${esc(m.sourcePerson.display_name)}" Into…</h3>
            <div class="spacer"></div>
            <button class="icon-btn" type="button" data-action="close-modal">${icon('close')}</button>
          </div>
          <div class="modal-body">
            <p class="muted" style="font-size:13.3px">
              All faces from <strong>${esc(m.sourcePerson.display_name)}</strong> will move into the selected target person and lock against future auto-splits.
            </p>
            <div class="search-box" style="margin-bottom:10px">
              ${icon('search')}
              <input type="search" id="modalPickerSearch" placeholder="Filter people…" value="${esc(m.search)}" />
            </div>
            <div class="picker-list" id="modalPickerList">
              ${renderPersonPickerRowsHtml(candidates, m.targetPersonId)}
            </div>
          </div>
          <div class="modal-foot">
            <button class="btn ghost" type="button" data-action="close-modal">Cancel</button>
            <button class="btn" type="button" data-action="submit-merge-person" ${!m.targetPersonId ? 'disabled' : ''}>Merge People</button>
          </div>
        </div>
      `;
    } else if (m.type === 'delete-person') {
      inner = `
        <div class="modal">
          <div class="modal-head">
            <h3>Delete "${esc(m.person.display_name)}"?</h3>
            <div class="spacer"></div>
            <button class="icon-btn" type="button" data-action="close-modal">${icon('close')}</button>
          </div>
          <div class="modal-body">
            <div class="note warn" style="margin-bottom:0">
              <strong>Original photos are never deleted.</strong><br/>
              This removes the person cluster "${esc(m.person.display_name)}" and moves its ${fmtNum(m.person.face_count)} face${m.person.face_count === 1 ? '' : 's'} back to <strong>Unknown Faces</strong>.
            </div>
          </div>
          <div class="modal-foot">
            <button class="btn ghost" type="button" data-action="close-modal">Cancel</button>
            <button class="btn danger" type="button" data-action="submit-delete-person">Delete Person Cluster</button>
          </div>
        </div>
      `;
    } else if (m.type === 'reassign-faces') {
      const candidates = state.people.filter(
        (p) =>
          p.id !== m.excludePersonId &&
          (!m.search || String(p.display_name).toLowerCase().includes(m.search.toLowerCase()))
      );
      inner = `
        <div class="modal">
          <div class="modal-head">
            <h3>Assign ${fmtNum(m.faceIds.length)} Face${m.faceIds.length === 1 ? '' : 's'} to Person</h3>
            <div class="spacer"></div>
            <button class="icon-btn" type="button" data-action="close-modal">${icon('close')}</button>
          </div>
          <div class="modal-body">
            <div class="search-box" style="margin-bottom:10px">
              ${icon('search')}
              <input type="search" id="modalPickerSearch" placeholder="Search target person…" value="${esc(m.search)}" />
            </div>
            <div class="picker-list" id="modalPickerList">
              ${renderPersonPickerRowsHtml(candidates, m.targetPersonId)}
            </div>
            <div style="margin-top:12px">
              <button class="btn ghost sm" type="button" data-action="switch-to-new-person-modal">
                ${icon('plus')}<span>Or create a brand new person instead…</span>
              </button>
            </div>
          </div>
          <div class="modal-foot">
            <button class="btn ghost" type="button" data-action="close-modal">Cancel</button>
            <button class="btn" type="button" data-action="submit-reassign-faces" ${!m.targetPersonId ? 'disabled' : ''}>Move Face${m.faceIds.length === 1 ? '' : 's'}</button>
          </div>
        </div>
      `;
    } else if (m.type === 'new-person') {
      inner = `
        <div class="modal">
          <div class="modal-head">
            <h3>Create New Person (${fmtNum(m.faceIds.length)} Face${m.faceIds.length === 1 ? '' : 's'})</h3>
            <div class="spacer"></div>
            <button class="icon-btn" type="button" data-action="close-modal">${icon('close')}</button>
          </div>
          <form id="modalNewPersonForm">
            <div class="modal-body">
              <div class="field" style="margin-bottom:0">
                <label>Person Name (optional)</label>
                <input type="text" id="modalNewPersonInput" placeholder="e.g. Priya, Dad, Alex…" value="${esc(m.displayName)}" maxlength="80" />
                <div class="hint">Leave blank to auto-assign a numbered label that you can rename anytime.</div>
              </div>
            </div>
            <div class="modal-foot">
              <button class="btn ghost" type="button" data-action="close-modal">Cancel</button>
              <button class="btn" type="submit">Create Person</button>
            </div>
          </form>
        </div>
      `;
    } else if (m.type === 'export') {
      const albums = (state.library && state.library.albums) || {};
      const peopleFiltered = state.people.filter(
        (p) => !m.search || String(p.display_name).toLowerCase().includes(m.search.toLowerCase())
      );

      inner = `
        <div class="modal wide">
          <div class="modal-head">
            <h3>Export Sorted Photo Copies</h3>
            <div class="spacer"></div>
            <button class="icon-btn" type="button" data-action="close-modal">${icon('close')}</button>
          </div>
          <div class="modal-body">
            <div class="field">
              <label>Destination Folder</label>
              <div class="btn-row" style="flex-wrap:nowrap">
                <input type="text" id="modalExportRootInput" value="${esc(m.exportRoot)}" placeholder="Choose where sorted folders will be created…" />
                <button class="btn ghost" type="button" data-action="modal-browse-export">Browse…</button>
              </div>
              <div class="hint">Source photos are copied safely — original files are never modified or deleted.</div>
            </div>

            <div class="field">
              <label>People Folders</label>
              <div class="seg" style="margin-bottom:10px">
                <button type="button" class="${m.scope === 'all' ? 'on' : ''}" data-action="set-export-scope" data-scope="all">All People (${fmtNum(state.people.length)})</button>
                <button type="button" class="${m.scope === 'selected' ? 'on' : ''}" data-action="set-export-scope" data-scope="selected">Selected People (${fmtNum(m.selectedPersonIds.size)})</button>
              </div>

              ${
                m.scope === 'selected'
                  ? `
                    <div style="display:flex;gap:8px;margin-bottom:8px;align-items:center">
                      <div class="search-box" style="flex:1">
                        ${icon('search')}
                        <input type="search" id="modalExportSearch" placeholder="Filter people…" value="${esc(m.search)}" />
                      </div>
                      <button class="btn ghost sm" type="button" data-action="export-select-all-people">All</button>
                      <button class="btn ghost sm" type="button" data-action="export-select-none-people">None</button>
                    </div>
                    <div class="picker-list" id="modalExportPeopleList" style="max-height:190px">
                      ${renderExportPeopleRowsHtml(peopleFiltered, m.selectedPersonIds)}
                    </div>
                  `
                  : ''
              }
            </div>

            <div class="field" style="margin-bottom:0">
              <label>Include Virtual Smart Albums</label>
              <div style="display:grid;gap:8px;margin-top:6px">
                <label style="display:flex;align-items:center;gap:9px;font-weight:450;cursor:pointer">
                  <input type="checkbox" id="expIncludeGroup" ${m.includeGroup ? 'checked' : ''} />
                  <span>Group Photos (${fmtNum(albums.group || 0)})</span>
                </label>
                <label style="display:flex;align-items:center;gap:9px;font-weight:450;cursor:pointer">
                  <input type="checkbox" id="expIncludeDocs" ${m.includeDocs ? 'checked' : ''} />
                  <span>Text &amp; Documents (${fmtNum(albums.docs || 0)})</span>
                </label>
                <label style="display:flex;align-items:center;gap:9px;font-weight:450;cursor:pointer">
                  <input type="checkbox" id="expIncludeUnknown" ${m.includeUnknown ? 'checked' : ''} />
                  <span>Unknown Faces (${fmtNum(albums.unknown || 0)})</span>
                </label>
              </div>
            </div>
          </div>
          <div class="modal-foot">
            <button class="btn ghost" type="button" data-action="close-modal">Cancel</button>
            <button class="btn" type="button" data-action="submit-export">
              ${icon('export')}<span>Start Safe Export</span>
            </button>
          </div>
        </div>
      `;
    }

    dom.modalHost.innerHTML = `<div class="modal-bg" data-action="modal-backdrop">${inner}</div>`;

    // Focus primary text input if present
    const autoInp =
      document.getElementById('modalRenameInput') ||
      document.getElementById('modalNewPersonInput') ||
      document.getElementById('modalPickerSearch');
    if (autoInp) {
      autoInp.focus();
      if (autoInp.id === 'modalRenameInput') autoInp.select();
    }
  }

  function renderPersonPickerRowsHtml(candidates, selectedId) {
    if (!candidates.length) {
      return `<div class="empty" style="padding:22px">No matching people</div>`;
    }
    return candidates
      .map((p) => {
        const sel = p.id === selectedId ? 'selected' : '';
        return `
          <div class="picker-row ${sel}" data-action="select-picker-person" data-person-id="${p.id}">
            ${
              p.cover_thumb_path
                ? `<img src="${esc(mediaSrc(p.cover_thumb_path))}" alt="" loading="lazy" />`
                : `<div class="pr-avatar">${icon('person')}</div>`
            }
            <div style="flex:1;min-width:0">
              <div class="truncate" style="font-weight:600;font-size:13.5px">${esc(p.display_name)}</div>
              <div class="muted tabular" style="font-size:12px">${fmtNum(p.photo_count)} photos · ${fmtNum(p.face_count)} faces</div>
            </div>
            ${sel ? `<span style="color:var(--green)">${icon('check')}</span>` : ''}
          </div>
        `;
      })
      .join('');
  }

  function renderExportPeopleRowsHtml(peopleList, selectedSet) {
    if (!peopleList.length) {
      return `<div class="empty" style="padding:18px">No people match filter</div>`;
    }
    return peopleList
      .map((p) => {
        const checked = selectedSet.has(p.id);
        return `
          <div class="picker-row ${checked ? 'selected' : ''}" data-action="toggle-export-person" data-person-id="${p.id}">
            <input type="checkbox" ${checked ? 'checked' : ''} style="pointer-events:none" />
            ${
              p.cover_thumb_path
                ? `<img src="${esc(mediaSrc(p.cover_thumb_path))}" alt="" loading="lazy" />`
                : `<div class="pr-avatar">${icon('person')}</div>`
            }
            <div style="flex:1;min-width:0">
              <div class="truncate" style="font-weight:550;font-size:13.3px">${esc(p.display_name)}</div>
            </div>
            <span class="muted tabular" style="font-size:12px">${fmtNum(p.photo_count)}</span>
          </div>
        `;
      })
      .join('');
  }

  // ---------------------------------------------------------------- Interactive Lightbox & Similarity Inspector

  async function openLightbox(imageId, contextList = [], contextIndex = 0, preselectFaceId = null) {
    state.lightbox.open = true;
    state.lightbox.loading = true;
    state.lightbox.imageId = imageId;
    state.lightbox.contextList = contextList;
    state.lightbox.contextIndex = contextIndex;
    state.lightbox.similarResult = null;
    state.lightbox.similarLoading = false;
    renderLightbox();

    try {
      const res = await api.rpc('photos.get', folderParams({ image_id: imageId }));
      state.lightbox.photo = res.photo;
      state.lightbox.faces = res.faces || [];
      state.lightbox.loading = false;

      if (preselectFaceId && state.lightbox.faces.some((f) => f.id === preselectFaceId)) {
        state.lightbox.selectedFaceId = preselectFaceId;
      } else if (state.lightbox.faces.length > 0) {
        state.lightbox.selectedFaceId = state.lightbox.faces[0].id;
      } else {
        state.lightbox.selectedFaceId = null;
      }
      renderLightbox();
    } catch (err) {
      toast(err.message || 'Could not load photo details', 'bad');
      closeLightbox();
    }
  }

  function closeLightbox() {
    if (lbResizeObserver) {
      lbResizeObserver.disconnect();
      lbResizeObserver = null;
    }
    state.lightbox.open = false;
    state.lightbox.photo = null;
    state.lightbox.faces = [];
    state.lightbox.selectedFaceId = null;
    state.lightbox.similarResult = null;
    dom.lightboxHost.innerHTML = '';
  }

  function stepLightbox(delta) {
    const lb = state.lightbox;
    if (!lb.open || !lb.contextList.length) return;
    const nextIdx = lb.contextIndex + delta;
    if (nextIdx < 0 || nextIdx >= lb.contextList.length) return;
    const target = lb.contextList[nextIdx];
    openLightbox(target.imageId, lb.contextList, nextIdx, target.faceId || null);
  }

  function renderLightbox() {
    if (lbResizeObserver) {
      lbResizeObserver.disconnect();
      lbResizeObserver = null;
    }
    const lb = state.lightbox;
    if (!lb.open) {
      dom.lightboxHost.innerHTML = '';
      return;
    }

    const photo = lb.photo;
    const hasPrev = lb.contextIndex > 0;
    const hasNext = lb.contextList.length > 0 && lb.contextIndex < lb.contextList.length - 1;
    const posLabel =
      lb.contextList.length > 1 ? `${lb.contextIndex + 1} / ${fmtNum(lb.contextList.length)}` : '';

    const selectedFace = lb.faces.find((f) => f.id === lb.selectedFaceId) || null;

    const sidebarHtml = lb.showSidebar
      ? `
        <aside class="lb-sidebar">
          <div class="lb-side-head">
            <h4>Photo &amp; Face Inspector</h4>
            <span class="pill tabular">${fmtNum(lb.faces.length)} face${lb.faces.length === 1 ? '' : 's'}</span>
          </div>
          <div class="lb-side-body" id="lbSideBody">
            ${renderLightboxSidebarBodyHtml(photo, lb.faces, selectedFace)}
          </div>
        </aside>
      `
      : '';

    dom.lightboxHost.innerHTML = `
      <div class="lightbox ${lb.showSidebar ? '' : 'no-sidebar'}">
        <div class="lb-stage-col">
          <div class="lb-topbar">
            <button class="lb-btn" type="button" data-action="close-lightbox" title="Close (Esc)">
              ${icon('arrowLeft')}<span>Back</span>
            </button>
            <div style="min-width:0;flex:1;margin-left:4px">
              <div class="lb-title">${esc(photo ? photo.filename : 'Loading…')}</div>
              <div class="lb-sub tabular">
                ${photo && photo.width ? `${photo.width}×${photo.height} · ${fmtBytes(photo.size)}` : ''}
                ${posLabel ? ` · ${posLabel}` : ''}
              </div>
            </div>

            <button class="lb-btn ${lb.showBoxes ? 'on' : ''}" type="button" data-action="toggle-lb-boxes" title="Toggle face boxes (B)">
              ${icon('target')}<span>Boxes</span>
            </button>
            ${
              photo
                ? `<button class="lb-btn" type="button" data-action="reveal-in-explorer" data-path="${esc(photo.path)}" title="Reveal original file in Explorer">
                     ${icon('external')}<span>Reveal</span>
                   </button>`
                : ''
            }
            <button class="lb-btn ${lb.showSidebar ? 'on' : ''}" type="button" data-action="toggle-lb-sidebar">
              <span>Inspector</span>
            </button>
            <button class="lb-btn" type="button" data-action="close-lightbox" title="Close (Esc)">
              ${icon('close')}
            </button>
          </div>

          <div class="lb-canvas" data-action="lb-canvas-bg">
            <button class="lb-nav-arrow prev" type="button" data-action="lb-prev" ${!hasPrev ? 'disabled' : ''} title="Previous (←)">
              ${icon('chevronLeft')}
            </button>

            ${
              lb.loading || !photo
                ? `<div style="color:rgba(255,255,255,0.7);font-size:14px">Loading high-resolution image…</div>`
                : `
                  <div class="lb-media-wrap" id="lbMediaWrap">
                    <img id="lbFullImage" src="${esc(mediaSrc(photo.path))}" alt="${esc(photo.filename)}" />
                    <div class="lb-overlay ${lb.showBoxes ? '' : 'hidden'}" id="lbOverlay"></div>
                  </div>
                `
            }

            <button class="lb-nav-arrow next" type="button" data-action="lb-next" ${!hasNext ? 'disabled' : ''} title="Next (→)">
              ${icon('chevronRight')}
            </button>
          </div>
        </div>

        ${sidebarHtml}
      </div>
    `;

    const imgEl = document.getElementById('lbFullImage');
    if (imgEl) {
      const syncBoxes = () => renderLightboxFaceBoxes();
      imgEl.addEventListener('load', syncBoxes);
      if (imgEl.complete && imgEl.naturalWidth > 0) {
        syncBoxes();
      }
      lbResizeObserver = new ResizeObserver(syncBoxes);
      lbResizeObserver.observe(imgEl);
    }
  }

  function renderLightboxFaceBoxes() {
    const lb = state.lightbox;
    const imgEl = document.getElementById('lbFullImage');
    const overlay = document.getElementById('lbOverlay');
    if (!imgEl || !overlay || !lb.photo) return;

    const dispW = imgEl.clientWidth;
    const dispH = imgEl.clientHeight;
    if (!dispW || !dispH) return;

    const baseW = lb.photo.width || imgEl.naturalWidth || dispW;
    const baseH = lb.photo.height || imgEl.naturalHeight || dispH;
    const sx = dispW / baseW;
    const sy = dispH / baseH;

    overlay.innerHTML = lb.faces
      .map((f, i) => {
        const [bx, by, bw, bh] = f.bbox || [0, 0, 0, 0];
        const left = Math.max(0, bx * sx);
        const top = Math.max(0, by * sy);
        const width = Math.max(18, bw * sx);
        const height = Math.max(18, bh * sy);
        const unassigned = !f.person_id ? 'unassigned' : '';
        const active = f.id === lb.selectedFaceId ? 'active' : '';
        const tagLabel = f.person_name || `Unknown #${i + 1}`;
        return `
          <div
            class="face-box ${unassigned} ${active}"
            data-action="select-lb-face"
            data-face-id="${f.id}"
            style="left:${left.toFixed(1)}px;top:${top.toFixed(1)}px;width:${width.toFixed(1)}px;height:${height.toFixed(1)}px"
          >
            <span class="face-box-tag">${esc(tagLabel)}</span>
          </div>
        `;
      })
      .join('');
  }

  function renderLightboxSidebarBodyHtml(photo, faces, selectedFace) {
    if (!photo) {
      return `<div class="muted">Loading metadata…</div>`;
    }

    const chipsHtml = faces.length
      ? `
        <div class="section-title">Faces in This Photo</div>
        <div class="face-chip-list">
          ${faces
            .map((f, idx) => {
              const active = selectedFace && selectedFace.id === f.id ? 'active' : '';
              const label = f.person_name || `Unknown #${idx + 1}`;
              return `
                <button class="face-chip ${active}" type="button" data-action="select-lb-face" data-face-id="${f.id}">
                  ${f.thumb_path ? `<img src="${esc(mediaSrc(f.thumb_path))}" alt="" />` : ''}
                  <span>${esc(label)}</span>
                </button>
              `;
            })
            .join('')}
        </div>
      `
      : `<div class="note" style="margin-bottom:14px">No faces detected in this photo.</div>`;

    let inspectorHtml = '';
    if (selectedFace) {
      const sim = state.lightbox.similarResult;
      let simHtml = '';
      if (state.lightbox.similarLoading) {
        simHtml = `<div class="muted" style="margin-top:10px;font-size:13px">Searching 128-D cosine similarity index…</div>`;
      } else if (sim && sim.query && sim.query.id === selectedFace.id) {
        if (!sim.matches.length) {
          simHtml = `<div class="muted" style="margin-top:10px;font-size:13px">No other embedded faces in library to compare.</div>`;
        } else {
          simHtml = `
            <div class="section-title">Closest Cosine Matches</div>
            <div class="similar-grid">
              ${sim.matches
                .map((m) => {
                  const pillCls = m.similarity >= 0.5 ? 'ok' : m.similarity >= 0.36 ? 'warn' : 'bad';
                  return `
                    <div class="sim-card" data-action="open-similar-match" data-image-id="${m.image_id}" data-face-id="${m.face_id}" title="${esc(m.person_name || 'Unknown')} · cosine ${m.similarity.toFixed(3)}">
                      <img src="${esc(mediaSrc(m.thumb_path || m.image_path))}" alt="" loading="lazy" />
                      <span class="pill ${pillCls} tabular" style="font-size:10.5px;padding:1px 6px">${m.similarity.toFixed(2)}</span>
                      <div class="truncate muted" style="font-size:11px;margin-top:3px">${esc(m.person_name || 'Unassigned')}</div>
                    </div>
                  `;
                })
                .join('')}
            </div>
          `;
        }
      }

      inspectorHtml = `
        <div class="card" style="padding:13px;background:var(--surface)">
          <div style="display:flex;align-items:center;gap:11px;margin-bottom:11px">
            ${
              selectedFace.thumb_path
                ? `<img src="${esc(mediaSrc(selectedFace.thumb_path))}" alt="" style="width:52px;height:52px;border-radius:10px;object-fit:cover;border:1px solid var(--line)" />`
                : ''
            }
            <div style="min-width:0;flex:1">
              <div style="font-weight:600;font-size:14.2px">${esc(selectedFace.person_name || 'Unassigned Face')}</div>
              <div class="muted tabular" style="font-size:12px">
                Conf ${(selectedFace.det_score * 100).toFixed(0)}% · Quality ${selectedFace.quality.toFixed(2)} · ${esc(selectedFace.assign_source)}
              </div>
            </div>
          </div>

          <div class="btn-row">
            <button class="btn sm" type="button" data-action="lb-assign-face" data-face-id="${selectedFace.id}">
              ${icon('people')}<span>${selectedFace.person_id ? 'Move to…' : 'Assign to…'}</span>
            </button>
            <button class="btn ghost sm" type="button" data-action="lb-new-person-face" data-face-id="${selectedFace.id}">
              ${icon('plus')}<span>New Person</span>
            </button>
            ${
              selectedFace.person_id
                ? `
                  <button class="btn ghost sm" type="button" data-action="set-person-cover" data-person-id="${selectedFace.person_id}" data-face-id="${selectedFace.id}">
                    ${icon('star')}<span>Set Cover</span>
                  </button>
                  <button class="btn danger sm" type="button" data-action="lb-unassign-face" data-face-id="${selectedFace.id}">
                    ${icon('unlink')}<span>Unassign</span>
                  </button>
                `
                : ''
            }
          </div>

          <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--line)">
            <button class="btn soft sm block" type="button" data-action="lb-find-similar" data-face-id="${selectedFace.id}">
              ${icon('similar')}<span>Find Similar Faces (Cosine Search)</span>
            </button>
            ${simHtml}
          </div>
        </div>
      `;
    }

    return `
      ${chipsHtml}
      ${inspectorHtml}

      <div class="section-title">File Metadata</div>
      <dl class="kv" style="grid-template-columns:92px 1fr;font-size:12.8px">
        <dt>Filename</dt>
        <dd>${esc(photo.filename)}</dd>
        <dt>Dimensions</dt>
        <dd class="tabular">${photo.width ? `${photo.width} × ${photo.height}` : '—'}</dd>
        <dt>Size</dt>
        <dd class="tabular">${fmtBytes(photo.size)}</dd>
        ${photo.taken_at ? `<dt>Captured</dt><dd class="tabular">${esc(photo.taken_at)}</dd>` : ''}
        <dt>Doc Score</dt>
        <dd class="tabular">${photo.doc_score}</dd>
        <dt>Path</dt>
        <dd class="mono" style="font-size:11.5px">${esc(photo.path)}</dd>
      </dl>
    `;
  }

  async function refreshOpenLightboxPhoto() {
    if (!state.lightbox.open || !state.lightbox.imageId) return;
    const prevFaceId = state.lightbox.selectedFaceId;
    const res = await api.rpc('photos.get', folderParams({ image_id: state.lightbox.imageId }));
    state.lightbox.photo = res.photo;
    state.lightbox.faces = res.faces || [];
    if (prevFaceId && state.lightbox.faces.some((f) => f.id === prevFaceId)) {
      state.lightbox.selectedFaceId = prevFaceId;
    } else {
      state.lightbox.selectedFaceId = state.lightbox.faces[0] ? state.lightbox.faces[0].id : null;
    }
    renderLightboxFaceBoxes();
    const sideBody = document.getElementById('lbSideBody');
    if (sideBody) {
      const selFace = state.lightbox.faces.find((f) => f.id === state.lightbox.selectedFaceId) || null;
      sideBody.innerHTML = renderLightboxSidebarBodyHtml(state.lightbox.photo, state.lightbox.faces, selFace);
    }
  }

  // ---------------------------------------------------------------- Event Delegation (Clicks, Inputs, Keys, Drag-Drop)

  function toggleFaceSelectionWithRange(faceId, idx, orderedFaceIds, shiftKey) {
    const idNum = Number(faceId);
    if (shiftKey && state.lastClickedFaceIndex !== null && orderedFaceIds.length > 0) {
      const start = Math.min(state.lastClickedFaceIndex, idx);
      const end = Math.max(state.lastClickedFaceIndex, idx);
      const shouldSelect = !state.selectedFaceIds.has(idNum);
      for (let i = start; i <= end; i++) {
        const fid = orderedFaceIds[i];
        if (fid !== undefined) {
          if (shouldSelect) state.selectedFaceIds.add(fid);
          else state.selectedFaceIds.delete(fid);
        }
      }
    } else if (state.selectedFaceIds.has(idNum)) {
      state.selectedFaceIds.delete(idNum);
    } else {
      state.selectedFaceIds.add(idNum);
    }
    state.lastClickedFaceIndex = idx;
    renderView();
  }

  async function commitInlineRename(personId, newNameRaw) {
    const trimmed = String(newNameRaw || '').trim();
    state.inlineRenameId = null;
    if (!trimmed) {
      renderView();
      return;
    }
    try {
      await api.rpc('people.rename', folderParams({ person_id: personId, new_name: trimmed }));
      await Promise.all([refreshPeopleList(), refreshLibraryInspect()]);
      renderApp();
      toast(`Renamed to "${trimmed}"`, 'ok', 2200);
    } catch (err) {
      toast(err.message || 'Could not rename person', 'bad');
      renderView();
    }
  }

  document.addEventListener('click', async (ev) => {
    // Commit inline rename if clicking outside input
    if (state.inlineRenameId !== null && !ev.target.closest('#inlineRenameInput') && !ev.target.closest('[data-action="start-inline-rename"]')) {
      const inp = document.getElementById('inlineRenameInput');
      if (inp) {
        await commitInlineRename(state.inlineRenameId, inp.value);
        return;
      }
      state.inlineRenameId = null;
    }

    const actionEl = ev.target.closest('[data-action]');
    if (!actionEl) return;

    const action = actionEl.dataset.action;

    // Prevent card click-through when clicking nested action buttons
    if (
      [
        'start-inline-rename',
        'open-merge-modal',
        'confirm-delete-person',
        'remove-recent',
        'toggle-face-select',
        'toggle-unknown-face-select',
        'set-person-cover',
        'quick-assign-single-face'
      ].includes(action)
    ) {
      ev.stopPropagation();
    }

    switch (action) {
      case 'pick-photo-folder': {
        ev.stopPropagation();
        const folder = await api.pickFolder({ title: 'Select Photo Library Folder' });
        if (folder) await openLibrary(folder, state.customDataFolder);
        break;
      }

      case 'toggle-welcome-custom-data': {
        state.useCustomDataOnWelcome = true;
        renderView();
        break;
      }

      case 'pick-welcome-data-folder': {
        const folder = await api.pickFolder({ title: 'Select Custom .autosorter Data Folder' });
        if (folder) {
          state.customDataFolder = folder;
          renderView();
        }
        break;
      }

      case 'open-recent': {
        const item = state.recentLibraries[Number(actionEl.dataset.idx)];
        if (item) await openLibrary(item.photoFolder, item.customDataFolder);
        break;
      }

      case 'remove-recent': {
        const idx = Number(actionEl.dataset.idx);
        state.recentLibraries.splice(idx, 1);
        savePrefs();
        renderView();
        break;
      }

      case 'set-speed': {
        const mode = actionEl.dataset.speed;
        if (['quiet', 'normal', 'fast'].includes(mode) && !state.pipeline.running) {
          state.speedMode = mode;
          savePrefs();
          renderView();
        }
        break;
      }

      case 'switch-tab': {
        const nextTab = actionEl.dataset.tab;
        if (!nextTab || !state.photoFolder) return;
        state.tab = nextTab;
        state.inlineRenameId = null;
        if (nextTab === 'people') {
          state.selectedPersonId = null;
          state.personDetail = null;
          state.selectedFaceIds.clear();
          await refreshPeopleList();
        } else if (nextTab === 'photos') {
          await refreshPhotosList();
        } else if (nextTab === 'albums') {
          state.selectedAlbumType = null;
          state.albumDetail = null;
          state.selectedFaceIds.clear();
          await refreshAlbumsSummary();
        } else if (nextTab === 'overview') {
          await refreshAllLibraryCollections();
        } else if (nextTab === 'settings') {
          try {
            state.sysInfo = await api.rpc('system.info', {});
          } catch (_e) {}
        }
        renderApp();
        break;
      }

      case 'jump-photos-filter': {
        state.tab = 'photos';
        state.photosFilter = actionEl.dataset.filter || 'all';
        state.photosPersonId = '';
        await refreshPhotosList();
        renderApp();
        break;
      }

      case 'pipeline-start':
        await startPipeline();
        break;

      case 'pipeline-cancel':
        await cancelPipeline();
        break;

      case 'open-in-explorer':
        if (actionEl.dataset.path) await api.openFolder(actionEl.dataset.path);
        break;

      case 'reveal-in-explorer':
        if (actionEl.dataset.path) await api.revealInExplorer(actionEl.dataset.path);
        break;

      case 'copy-text':
        await copyText(actionEl.dataset.text, actionEl.dataset.label || 'Path');
        break;

      case 'open-switch-library':
        openSwitchLibraryModal();
        break;

      // ---------------- People & Person Detail
      case 'open-person': {
        if (ev.target.closest('#inlineRenameInput')) return;
        const pid = Number(actionEl.dataset.personId);
        state.tab = 'people';
        state.selectedPersonId = pid;
        state.selectedFaceIds.clear();
        state.lastClickedFaceIndex = null;
        await refreshPersonDetail(pid);
        renderApp();
        break;
      }

      case 'close-person-detail': {
        state.selectedPersonId = null;
        state.personDetail = null;
        state.selectedFaceIds.clear();
        await refreshPeopleList();
        renderApp();
        break;
      }

      case 'start-inline-rename': {
        state.inlineRenameId = Number(actionEl.dataset.personId);
        renderView();
        break;
      }

      case 'open-rename-modal':
        openRenameModal(Number(actionEl.dataset.personId));
        break;

      case 'open-merge-modal':
        openMergeModal(Number(actionEl.dataset.personId));
        break;

      case 'confirm-delete-person':
        openDeletePersonModal(Number(actionEl.dataset.personId));
        break;

      case 'recheck-person-album': {
        const pid = Number(actionEl.dataset.personId);
        try {
          const res = await api.rpc('people.recheck', folderParams({ person_id: pid }));
          await refreshAllLibraryCollections();
          renderApp();
          toast(
            res.total_shifted > 0
              ? `Active learning shifted ${res.total_shifted} borderline face${res.total_shifted === 1 ? '' : 's'}`
              : 'Album verified — all faces match cleanly',
            'ok'
          );
        } catch (err) {
          toast(err.message || 'Could not re-check album', 'bad');
        }
        break;
      }

      case 'recluster-auto-faces': {
        try {
          const res = await api.rpc('people.recluster', folderParams());
          await refreshAllLibraryCollections();
          renderApp();
          toast(`Re-clustered ${fmtNum(res.faces_assigned)} faces into ${fmtNum(res.persons_count)} people`, 'ok');
        } catch (err) {
          toast(err.message || 'Could not re-cluster faces', 'bad');
        }
        break;
      }

      case 'set-person-cover': {
        const pid = Number(actionEl.dataset.personId);
        const fid = Number(actionEl.dataset.faceId);
        try {
          await api.rpc('people.set_cover', folderParams({ person_id: pid, face_id: fid }));
          await refreshAllLibraryCollections();
          renderApp();
          toast('Updated cover photo', 'ok', 2000);
        } catch (err) {
          toast(err.message || 'Could not update cover photo', 'bad');
        }
        break;
      }

      case 'toggle-face-select': {
        const faces = (state.personDetail && state.personDetail.faces) || [];
        toggleFaceSelectionWithRange(
          Number(actionEl.dataset.faceId),
          Number(actionEl.dataset.idx),
          faces.map((f) => f.id),
          ev.shiftKey
        );
        break;
      }

      case 'select-all-faces': {
        const faces = (state.personDetail && state.personDetail.faces) || [];
        if (state.selectedFaceIds.size === faces.length) {
          state.selectedFaceIds.clear();
        } else {
          faces.forEach((f) => state.selectedFaceIds.add(f.id));
        }
        renderView();
        break;
      }

      case 'clear-face-selection': {
        state.selectedFaceIds.clear();
        renderView();
        break;
      }

      case 'open-reassign-faces-modal':
        openReassignFacesModal();
        break;

      case 'open-new-person-modal':
        openNewPersonModal();
        break;

      case 'batch-unassign-faces': {
        const ids = Array.from(state.selectedFaceIds);
        if (!ids.length) return;
        try {
          await api.rpc('faces.unassign', folderParams({ face_ids: ids }));
          state.selectedFaceIds.clear();
          await refreshAllLibraryCollections();
          renderApp();
          toast(`Moved ${ids.length} face${ids.length === 1 ? '' : 's'} to Unknown Faces`, 'ok');
        } catch (err) {
          toast(err.message || 'Could not unassign faces', 'bad');
        }
        break;
      }

      case 'filter-photos-by-person': {
        state.tab = 'photos';
        state.photosFilter = 'all';
        state.photosPersonId = String(actionEl.dataset.personId || '');
        await refreshPhotosList();
        renderApp();
        break;
      }

      case 'open-person-face-lightbox': {
        const faces = (state.personDetail && state.personDetail.faces) || [];
        const idx = Number(actionEl.dataset.idx);
        const f = faces[idx];
        if (!f) return;
        const ctx = faces.map((item) => ({ imageId: item.image_id, faceId: item.id }));
        await openLightbox(f.image_id, ctx, idx, f.id);
        break;
      }

      // ---------------- Photos Tab
      case 'set-photos-filter': {
        state.photosFilter = actionEl.dataset.filter || 'all';
        await refreshPhotosList();
        renderView();
        break;
      }

      case 'set-grid-size': {
        const sz = actionEl.dataset.size;
        if (GRID_SIZES[sz]) {
          state.photoGridSize = sz;
          savePrefs();
          renderView();
        }
        break;
      }

      case 'open-grid-photo': {
        const idx = Number(actionEl.dataset.idx);
        const photo = state.photos[idx];
        if (!photo) return;
        const ctx = state.photos.map((p) => ({ imageId: p.id }));
        await openLightbox(photo.id, ctx, idx);
        break;
      }

      // ---------------- Smart Albums & Export
      case 'open-album': {
        const albumType = actionEl.dataset.album;
        state.tab = 'albums';
        state.selectedAlbumType = albumType;
        state.selectedFaceIds.clear();
        state.lastClickedFaceIndex = null;
        await refreshAlbumDetail(albumType);
        renderApp();
        break;
      }

      case 'close-album-detail': {
        state.selectedAlbumType = null;
        state.albumDetail = null;
        state.selectedFaceIds.clear();
        await refreshAlbumsSummary();
        renderApp();
        break;
      }

      case 'toggle-unknown-face-select': {
        const items = (state.albumDetail && state.albumDetail.items) || [];
        toggleFaceSelectionWithRange(
          Number(actionEl.dataset.faceId),
          Number(actionEl.dataset.idx),
          items.map((it) => it.face_id),
          ev.shiftKey
        );
        break;
      }

      case 'select-all-unknown-faces': {
        const items = (state.albumDetail && state.albumDetail.items) || [];
        if (state.selectedFaceIds.size === items.length) {
          state.selectedFaceIds.clear();
        } else {
          items.forEach((it) => state.selectedFaceIds.add(it.face_id));
        }
        renderView();
        break;
      }

      case 'quick-assign-single-face': {
        const fid = Number(actionEl.dataset.faceId);
        if (state.people.length > 0) {
          openReassignFacesModal([fid]);
        } else {
          openNewPersonModal([fid]);
        }
        break;
      }

      case 'open-album-lightbox': {
        const items = (state.albumDetail && state.albumDetail.items) || [];
        const idx = Number(actionEl.dataset.idx);
        const item = items[idx];
        if (!item) return;
        const ctx = items.map((it) => ({ imageId: it.image_id, faceId: it.face_id || null }));
        await openLightbox(item.image_id, ctx, idx, item.face_id || null);
        break;
      }

      case 'open-export-modal':
        openExportModal();
        break;

      case 'open-export-person':
        openExportModal([Number(actionEl.dataset.personId)]);
        break;

      case 'export-cancel': {
        state.exportJob.cancelling = true;
        renderView();
        await api.rpc('export.cancel', {});
        break;
      }

      // ---------------- Modal Actions
      case 'modal-backdrop':
        if (ev.target === actionEl) closeModal();
        break;

      case 'close-modal':
        closeModal();
        break;

      case 'modal-browse-photo': {
        const p = await api.pickFolder({ title: 'Select Photo Library Folder' });
        if (p && state.modal) {
          state.modal.photoFolder = p;
          const inp = document.getElementById('modalPhotoFolderInput');
          if (inp) inp.value = p;
        }
        break;
      }

      case 'modal-browse-data': {
        const p = await api.pickFolder({ title: 'Select Custom .autosorter Data Folder' });
        if (p && state.modal) {
          state.modal.customDataFolder = p;
          const inp = document.getElementById('modalCustomDataInput');
          if (inp) inp.value = p;
        }
        break;
      }

      case 'modal-open-recent': {
        const r = state.recentLibraries[Number(actionEl.dataset.idx)];
        if (r) {
          closeModal();
          await openLibrary(r.photoFolder, r.customDataFolder);
        }
        break;
      }

      case 'submit-switch-library': {
        const pInp = document.getElementById('modalPhotoFolderInput');
        const dInp = document.getElementById('modalCustomDataInput');
        const photoVal = pInp ? pInp.value.trim() : '';
        const dataVal = dInp ? dInp.value.trim() : '';
        if (!photoVal) {
          toast('Choose a photo folder to open', 'bad');
          return;
        }
        closeModal();
        await openLibrary(photoVal, dataVal || null);
        break;
      }

      case 'select-picker-person': {
        if (!state.modal) return;
        state.modal.targetPersonId = Number(actionEl.dataset.personId);
        renderModal();
        break;
      }

      case 'submit-merge-person': {
        const m = state.modal;
        if (!m || !m.targetPersonId) return;
        try {
          const targetId = m.targetPersonId;
          await api.rpc(
            'people.merge',
            folderParams({
              source_person_id: m.sourcePerson.id,
              target_person_id: targetId
            })
          );
          closeModal();
          if (state.selectedPersonId === m.sourcePerson.id) {
            state.selectedPersonId = targetId;
          }
          await refreshAllLibraryCollections();
          renderApp();
          toast('Merged people clusters cleanly', 'ok');
        } catch (err) {
          toast(err.message || 'Merge failed', 'bad');
        }
        break;
      }

      case 'submit-delete-person': {
        const m = state.modal;
        if (!m || !m.person) return;
        try {
          await api.rpc('people.delete', folderParams({ person_id: m.person.id }));
          if (state.selectedPersonId === m.person.id) {
            state.selectedPersonId = null;
            state.personDetail = null;
          }
          closeModal();
          await refreshAllLibraryCollections();
          renderApp();
          toast(`Deleted "${m.person.display_name}"`, 'ok');
        } catch (err) {
          toast(err.message || 'Could not delete person', 'bad');
        }
        break;
      }

      case 'switch-to-new-person-modal': {
        if (state.modal && state.modal.faceIds) {
          openNewPersonModal(state.modal.faceIds);
        }
        break;
      }

      case 'submit-reassign-faces': {
        const m = state.modal;
        if (!m || !m.targetPersonId) return;
        try {
          await api.rpc(
            'faces.reassign',
            folderParams({ face_ids: m.faceIds, target_person_id: m.targetPersonId })
          );
          closeModal();
          state.selectedFaceIds.clear();
          await refreshAllLibraryCollections();
          await refreshOpenLightboxPhoto();
          renderApp();
          toast(`Assigned ${m.faceIds.length} face${m.faceIds.length === 1 ? '' : 's'}`, 'ok');
        } catch (err) {
          toast(err.message || 'Could not reassign faces', 'bad');
        }
        break;
      }

      case 'modal-browse-export': {
        const p = await api.pickFolder({
          title: 'Select Export Destination Folder',
          defaultPath: (state.modal && state.modal.exportRoot) || undefined
        });
        if (p && state.modal) {
          state.modal.exportRoot = p;
          const inp = document.getElementById('modalExportRootInput');
          if (inp) inp.value = p;
        }
        break;
      }

      case 'set-export-scope': {
        if (!state.modal) return;
        state.modal.scope = actionEl.dataset.scope;
        renderModal();
        break;
      }

      case 'toggle-export-person': {
        if (!state.modal) return;
        const pid = Number(actionEl.dataset.personId);
        if (state.modal.selectedPersonIds.has(pid)) state.modal.selectedPersonIds.delete(pid);
        else state.modal.selectedPersonIds.add(pid);
        renderModal();
        break;
      }

      case 'export-select-all-people': {
        if (!state.modal) return;
        state.people.forEach((p) => state.modal.selectedPersonIds.add(p.id));
        renderModal();
        break;
      }

      case 'export-select-none-people': {
        if (!state.modal) return;
        state.modal.selectedPersonIds.clear();
        renderModal();
        break;
      }

      case 'submit-export': {
        const m = state.modal;
        if (!m) return;
        const rootInp = document.getElementById('modalExportRootInput');
        const exportRoot = rootInp ? rootInp.value.trim() : m.exportRoot;
        if (!exportRoot) {
          toast('Choose an export destination folder first', 'bad');
          return;
        }
        const includeGroup = Boolean(document.getElementById('expIncludeGroup')?.checked);
        const includeDocs = Boolean(document.getElementById('expIncludeDocs')?.checked);
        const includeUnknown = Boolean(document.getElementById('expIncludeUnknown')?.checked);
        const personIds = m.scope === 'selected' ? Array.from(m.selectedPersonIds) : null;

        try {
          state.lastExportFolder = exportRoot;
          savePrefs();
          await api.rpc(
            'export.start',
            folderParams({
              export_root: exportRoot,
              person_ids: personIds,
              include_group: includeGroup,
              include_docs: includeDocs,
              include_unknown: includeUnknown
            })
          );
          state.exportJob.running = true;
          state.exportJob.cancelling = false;
          state.exportJob.done = 0;
          state.exportJob.total = 0;
          state.exportJob.detail = 'Preparing export…';
          state.exportJob.exportRoot = exportRoot;
          state.exportJob.lastResult = null;
          closeModal();
          state.tab = 'albums';
          state.selectedAlbumType = null;
          renderApp();
          toast('Started safe copy export', 'ok');
        } catch (err) {
          toast(err.message || 'Could not start export', 'bad');
        }
        break;
      }

      // ---------------- Lightbox Actions
      case 'close-lightbox':
        closeLightbox();
        break;

      case 'lb-canvas-bg':
        if (ev.target === actionEl) closeLightbox();
        break;

      case 'lb-prev':
        stepLightbox(-1);
        break;

      case 'lb-next':
        stepLightbox(1);
        break;

      case 'toggle-lb-boxes':
        state.lightbox.showBoxes = !state.lightbox.showBoxes;
        renderLightbox();
        break;

      case 'toggle-lb-sidebar':
        state.lightbox.showSidebar = !state.lightbox.showSidebar;
        renderLightbox();
        break;

      case 'select-lb-face': {
        ev.stopPropagation();
        state.lightbox.selectedFaceId = Number(actionEl.dataset.faceId);
        renderLightboxFaceBoxes();
        const sideBody = document.getElementById('lbSideBody');
        if (sideBody) {
          const sel = state.lightbox.faces.find((f) => f.id === state.lightbox.selectedFaceId) || null;
          sideBody.innerHTML = renderLightboxSidebarBodyHtml(state.lightbox.photo, state.lightbox.faces, sel);
        }
        break;
      }

      case 'lb-assign-face': {
        const fid = Number(actionEl.dataset.faceId);
        if (state.people.length > 0) openReassignFacesModal([fid]);
        else openNewPersonModal([fid]);
        break;
      }

      case 'lb-new-person-face':
        openNewPersonModal([Number(actionEl.dataset.faceId)]);
        break;

      case 'lb-unassign-face': {
        const fid = Number(actionEl.dataset.faceId);
        try {
          await api.rpc('faces.unassign', folderParams({ face_id: fid }));
          await refreshAllLibraryCollections();
          await refreshOpenLightboxPhoto();
          renderApp();
          toast('Face moved to Unknown Faces', 'ok');
        } catch (err) {
          toast(err.message || 'Could not unassign face', 'bad');
        }
        break;
      }

      case 'lb-find-similar': {
        const fid = Number(actionEl.dataset.faceId);
        state.lightbox.similarLoading = true;
        const sideBody = document.getElementById('lbSideBody');
        if (sideBody) {
          const sel = state.lightbox.faces.find((f) => f.id === state.lightbox.selectedFaceId) || null;
          sideBody.innerHTML = renderLightboxSidebarBodyHtml(state.lightbox.photo, state.lightbox.faces, sel);
        }
        try {
          const sim = await api.rpc('faces.similar', folderParams({ face_id: fid, top_k: 9 }));
          state.lightbox.similarResult = sim;
        } catch (err) {
          toast(err.message || 'Similarity search failed', 'bad');
        } finally {
          state.lightbox.similarLoading = false;
          const sb = document.getElementById('lbSideBody');
          if (sb) {
            const sel = state.lightbox.faces.find((f) => f.id === state.lightbox.selectedFaceId) || null;
            sb.innerHTML = renderLightboxSidebarBodyHtml(state.lightbox.photo, state.lightbox.faces, sel);
          }
        }
        break;
      }

      case 'open-similar-match': {
        const imgId = Number(actionEl.dataset.imageId);
        const fid = Number(actionEl.dataset.faceId);
        await openLightbox(imgId, state.lightbox.contextList, state.lightbox.contextIndex, fid);
        break;
      }

      default:
        break;
    }
  });

  // Form submissions (Rename & New Person modals)
  document.addEventListener('submit', async (ev) => {
    if (ev.target.id === 'modalRenameForm') {
      ev.preventDefault();
      const inp = document.getElementById('modalRenameInput');
      const m = state.modal;
      if (!inp || !m) return;
      const newName = inp.value.trim();
      if (!newName) return;
      try {
        await api.rpc('people.rename', folderParams({ person_id: m.personId, new_name: newName }));
        closeModal();
        await refreshAllLibraryCollections();
        renderApp();
        toast(`Renamed to "${newName}"`, 'ok');
      } catch (err) {
        toast(err.message || 'Could not rename person', 'bad');
      }
    } else if (ev.target.id === 'modalNewPersonForm') {
      ev.preventDefault();
      const inp = document.getElementById('modalNewPersonInput');
      const m = state.modal;
      if (!m) return;
      const displayName = inp ? inp.value.trim() : '';
      try {
        const res = await api.rpc(
          'faces.create_person',
          folderParams({ face_ids: m.faceIds, display_name: displayName || null })
        );
        closeModal();
        state.selectedFaceIds.clear();
        await refreshAllLibraryCollections();
        await refreshOpenLightboxPhoto();
        renderApp();
        toast(`Created "${res.display_name}"`, 'ok');
      } catch (err) {
        toast(err.message || 'Could not create person', 'bad');
      }
    }
  });

  // Live inputs & selects (non-destructive focus preservation)
  document.addEventListener('input', (ev) => {
    const t = ev.target;
    if (t.id === 'peopleSearchInput') {
      state.peopleSearch = t.value;
      const host = document.getElementById('peopleGridHost');
      if (host) host.innerHTML = renderPeopleGridInnerHtml();
    } else if (t.id === 'photosSearchInput') {
      state.photosSearch = t.value;
      clearTimeout(photoSearchTimer);
      photoSearchTimer = setTimeout(async () => {
        await refreshPhotosList();
        if (state.tab === 'photos') mountVirtualPhotosGrid();
      }, 160);
    } else if (t.id === 'modalPickerSearch' && state.modal) {
      state.modal.search = t.value;
      const listEl = document.getElementById('modalPickerList');
      if (listEl) {
        const exclude =
          state.modal.type === 'merge-person' ? state.modal.sourcePerson.id : state.modal.excludePersonId;
        const cands = state.people.filter(
          (p) =>
            p.id !== exclude &&
            (!state.modal.search || String(p.display_name).toLowerCase().includes(state.modal.search.toLowerCase()))
        );
        if (cands.length && !cands.some((c) => c.id === state.modal.targetPersonId)) {
          state.modal.targetPersonId = cands[0].id;
        }
        listEl.innerHTML = renderPersonPickerRowsHtml(cands, state.modal.targetPersonId);
      }
    } else if (t.id === 'modalExportSearch' && state.modal) {
      state.modal.search = t.value;
      const listEl = document.getElementById('modalExportPeopleList');
      if (listEl) {
        const filtered = state.people.filter(
          (p) => !state.modal.search || String(p.display_name).toLowerCase().includes(state.modal.search.toLowerCase())
        );
        listEl.innerHTML = renderExportPeopleRowsHtml(filtered, state.modal.selectedPersonIds);
      }
    } else if (t.id === 'modalExportRootInput' && state.modal) {
      state.modal.exportRoot = t.value;
    }
  });

  document.addEventListener('change', async (ev) => {
    const t = ev.target;
    if (t.id === 'peopleSortSelect') {
      state.peopleSort = t.value;
      const host = document.getElementById('peopleGridHost');
      if (host) host.innerHTML = renderPeopleGridInnerHtml();
    } else if (t.id === 'photosPersonSelect') {
      state.photosPersonId = t.value;
      await refreshPhotosList();
      if (state.tab === 'photos') mountVirtualPhotosGrid();
    }
  });

  // Keyboard shortcuts & inline rename Enter/Escape
  document.addEventListener('keydown', async (ev) => {
    if (ev.target && ev.target.id === 'inlineRenameInput') {
      if (ev.key === 'Enter') {
        ev.preventDefault();
        await commitInlineRename(Number(ev.target.dataset.personId), ev.target.value);
      } else if (ev.key === 'Escape') {
        ev.preventDefault();
        state.inlineRenameId = null;
        renderView();
      }
      return;
    }

    if (ev.key === 'Escape') {
      if (state.modal) {
        ev.preventDefault();
        closeModal();
        return;
      }
      if (state.lightbox.open) {
        ev.preventDefault();
        closeLightbox();
        return;
      }
    }

    const inEditable = ev.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(ev.target.tagName);
    if (inEditable) return;

    if (state.lightbox.open && !state.modal) {
      if (ev.key === 'ArrowLeft') {
        ev.preventDefault();
        stepLightbox(-1);
      } else if (ev.key === 'ArrowRight') {
        ev.preventDefault();
        stepLightbox(1);
      } else if (ev.key === 'b' || ev.key === 'B') {
        ev.preventDefault();
        state.lightbox.showBoxes = !state.lightbox.showBoxes;
        renderLightbox();
      }
    }
  });

  // Brand click returns to Overview (or stays on Welcome)
  dom.brandBtn.addEventListener('click', async () => {
    if (!state.photoFolder) return;
    state.tab = 'overview';
    await refreshAllLibraryCollections();
    renderApp();
  });

  // ---------------------------------------------------------------- Global Folder Drag & Drop

  let dragDepth = 0;

  window.addEventListener('dragenter', (ev) => {
    if (!ev.dataTransfer || !Array.from(ev.dataTransfer.types || []).includes('Files')) return;
    ev.preventDefault();
    dragDepth++;
    dom.dropOverlay.classList.remove('hidden');
  });

  window.addEventListener('dragover', (ev) => {
    if (!ev.dataTransfer || !Array.from(ev.dataTransfer.types || []).includes('Files')) return;
    ev.preventDefault();
    ev.dataTransfer.dropEffect = 'copy';
  });

  window.addEventListener('dragleave', (ev) => {
    if (!ev.dataTransfer || !Array.from(ev.dataTransfer.types || []).includes('Files')) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      dom.dropOverlay.classList.add('hidden');
    }
  });

  window.addEventListener('drop', async (ev) => {
    ev.preventDefault();
    dragDepth = 0;
    dom.dropOverlay.classList.add('hidden');
    const files = ev.dataTransfer && ev.dataTransfer.files;
    if (!files || !files.length) return;
    const folderPath = await api.resolveDroppedPath(files[0]);
    if (folderPath) {
      closeModal();
      closeLightbox();
      await openLibrary(folderPath, null);
      toast(`Opened ${baseName(folderPath)}`, 'ok', 2400);
    } else {
      toast('Drop a local folder to open it in Auto-Sorter', 'bad');
    }
  });

  // ---------------------------------------------------------------- Python Sidecar Events

  api.onEvent(async (msg) => {
    const { event, data } = msg || {};
    if (!event) return;

    switch (event) {
      case 'bridge.ready':
        state.bridgeConnected = true;
        try {
          state.sysInfo = await api.rpc('system.info', {});
        } catch (_e) {}
        break;

      case 'bridge.disconnected':
        state.bridgeConnected = false;
        state.pipeline.running = false;
        state.exportJob.running = false;
        renderApp();
        break;

      case 'pipeline.stage':
        state.pipeline.running = true;
        state.pipeline.stage_num = data.stage_num || 1;
        state.pipeline.total_stages = data.total_stages || 4;
        state.pipeline.stage_name = data.stage_name || '';
        state.pipeline.done = 0;
        state.pipeline.total = 0;
        state.pipeline.detail = `${data.stage_name}…`;
        updatePipelineDomInPlace();
        // Refresh stats between stages so user sees live counts climb!
        refreshAllLibraryCollections().then(() => {
          renderTopbar();
        });
        break;

      case 'pipeline.progress':
        state.pipeline.running = true;
        state.pipeline.done = data.done || 0;
        state.pipeline.total = data.total || 0;
        state.pipeline.detail = data.detail || '';
        state.pipeline.stage_num = data.stage_num || state.pipeline.stage_num;
        state.pipeline.stage_name = data.stage_name || state.pipeline.stage_name;
        state.pipeline.rss_mb = data.rss_mb || state.pipeline.rss_mb;
        updatePipelineDomInPlace();
        break;

      case 'pipeline.finished':
        state.pipeline.running = false;
        state.pipeline.cancelling = false;
        state.pipeline.lastSummary = data || {};
        await refreshAllLibraryCollections();
        renderApp();
        toast(
          data && data.cancelled ? 'Scan paused — progress saved' : 'Library scan complete!',
          data && data.cancelled ? 'default' : 'ok'
        );
        break;

      case 'pipeline.error':
        state.pipeline.running = false;
        state.pipeline.cancelling = false;
        state.pipeline.lastError = (data && data.message) || 'Unknown pipeline error';
        await refreshAllLibraryCollections();
        renderApp();
        toast(`Scan error: ${state.pipeline.lastError}`, 'bad', 5000);
        break;

      case 'export.progress': {
        state.exportJob.running = true;
        state.exportJob.done = data.done || 0;
        state.exportJob.total = data.total || 0;
        state.exportJob.detail = data.detail || '';
        const txt = document.getElementById('exportLiveText');
        const fill = document.getElementById('exportLiveFill');
        if (txt) txt.textContent = state.exportJob.detail;
        if (fill && state.exportJob.total > 0) {
          fill.style.width = `${Math.min(100, Math.round((state.exportJob.done / state.exportJob.total) * 100))}%`;
        }
        break;
      }

      case 'export.finished':
        state.exportJob.running = false;
        state.exportJob.cancelling = false;
        state.exportJob.lastResult = data || {};
        renderApp();
        toast(
          data && data.cancelled
            ? 'Export cancelled'
            : `Exported ${fmtNum(data?.copied || 0)} photos cleanly`,
          data && data.cancelled ? 'default' : 'ok'
        );
        break;

      case 'export.error':
        state.exportJob.running = false;
        state.exportJob.cancelling = false;
        state.exportJob.lastError = (data && data.message) || 'Export error';
        renderApp();
        toast(`Export failed: ${state.exportJob.lastError}`, 'bad', 5000);
        break;

      default:
        break;
    }
  });

  // ---------------------------------------------------------------- Boot

  async function init() {
    if (api.platform === 'win32') {
      document.body.classList.add('is-win');
    }
    dom.brandMark.innerHTML = icon('brand');
    dom.dropOverlayIcon.innerHTML = icon('folderOpen');

    // Render immediately so window opens instantaneously with zero blank flash
    renderApp();

    try {
      state.sysInfo = await api.rpc('system.info', {});
      const status = await api.rpc('pipeline.status', {});
      if (status && status.pipeline && status.pipeline.running) {
        state.pipeline.running = true;
        Object.assign(state.pipeline, status.pipeline);
      }
      renderApp();
    } catch (_err) {
      // Sidecar may still be warming up; bridge.ready event will hydrate sysInfo
    }
  }

  init();
})();
