'use strict';

const { app, BrowserWindow, dialog, ipcMain, net, protocol, shell } = require('electron');
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { pathToFileURL } = require('url');

const IS_DEV = process.argv.includes('--dev');
const IS_SELFTEST = process.argv.includes('--selftest');
const REPO_ROOT = path.resolve(__dirname, '..');

// Allowed image extensions for the secure media protocol
const NATIVE_IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']);
const CONVERTED_IMAGE_EXTS = new Set([
  '.heic', '.heif', '.tif', '.tiff',
  '.cr2', '.cr3', '.nef', '.nrw', '.arw', '.srf', '.sr2',
  '.dng', '.raf', '.orf', '.rw2', '.pef'
]);

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'autosorter-media',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true
    }
  }
]);

// ---------------------------------------------------------------- Python Bridge Manager

let bridgeProc = null;
let bridgeReady = false;
let nextRpcId = 1;
const pendingRpc = new Map();
let mainWindow = null;
let isQuitting = false;

function resolveEngineCommand() {
  if (process.env.AUTOSORTER_ENGINE && fs.existsSync(process.env.AUTOSORTER_ENGINE)) {
    return {
      cmd: process.env.AUTOSORTER_ENGINE,
      args: [],
      cwd: path.dirname(process.env.AUTOSORTER_ENGINE)
    };
  }

  if (app.isPackaged) {
    const packagedExe = path.join(process.resourcesPath, 'engine', 'autosorter-engine.exe');
    return {
      cmd: packagedExe,
      args: [],
      cwd: path.dirname(packagedExe)
    };
  }

  if (process.env.AUTOSORTER_PYTHON && fs.existsSync(process.env.AUTOSORTER_PYTHON)) {
    return {
      cmd: process.env.AUTOSORTER_PYTHON,
      args: ['-u', '-m', 'autosorter.bridge'],
      cwd: REPO_ROOT
    };
  }

  const venvCandidates = [
    path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'),
    path.join(REPO_ROOT, '.venv', 'bin', 'python')
  ];
  for (const candidate of venvCandidates) {
    if (fs.existsSync(candidate)) {
      return { cmd: candidate, args: ['-u', '-m', 'autosorter.bridge'], cwd: REPO_ROOT };
    }
  }

  const probes = [
    { cmd: 'python', args: [] },
    { cmd: 'python3', args: [] },
    { cmd: 'py', args: ['-3'] }
  ];
  for (const probe of probes) {
    try {
      const res = spawnSync(probe.cmd, [...probe.args, '--version'], {
        windowsHide: true,
        timeout: 3000
      });
      if (res.status === 0) {
        return {
          cmd: probe.cmd,
          args: [...probe.args, '-u', '-m', 'autosorter.bridge'],
          cwd: REPO_ROOT
        };
      }
    } catch (_err) {
      // try next candidate
    }
  }

  const localFrozen = path.join(
    REPO_ROOT,
    'build',
    'bridge-dist',
    'autosorter-engine',
    'autosorter-engine.exe'
  );
  if (fs.existsSync(localFrozen)) {
    return { cmd: localFrozen, args: [], cwd: path.dirname(localFrozen) };
  }

  return { cmd: 'python', args: ['-u', '-m', 'autosorter.bridge'], cwd: REPO_ROOT };
}

function startPythonBridge() {
  if (bridgeProc) return;

  const engine = resolveEngineCommand();

  bridgeProc = spawn(engine.cmd, engine.args, {
    cwd: engine.cwd,
    windowsHide: true,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8'
    }
  });

  const rl = readline.createInterface({
    input: bridgeProc.stdout,
    crlfDelay: Infinity
  });

  rl.on('line', (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let msg;
    try {
      msg = JSON.parse(trimmed);
    } catch (_err) {
      return;
    }

    if (msg.id !== undefined && pendingRpc.has(msg.id)) {
      const { resolve, reject } = pendingRpc.get(msg.id);
      pendingRpc.delete(msg.id);
      if (msg.ok) {
        resolve(msg.result);
      } else {
        const err = new Error((msg.error && msg.error.message) || 'Python bridge error');
        err.code = (msg.error && msg.error.code) || 'BRIDGE_ERROR';
        reject(err);
      }
      return;
    }

    if (msg.event) {
      if (msg.event === 'bridge.ready') {
        bridgeReady = true;
      }
      broadcastEvent(msg);
    }
  });

  bridgeProc.stderr.on('data', (chunk) => {
    if (IS_DEV) {
      process.stderr.write(`[python] ${chunk.toString('utf8')}`);
    }
  });

  bridgeProc.on('exit', (code, signal) => {
    bridgeProc = null;
    bridgeReady = false;
    for (const [, { reject }] of pendingRpc) {
      reject(new Error(`Python engine exited unexpectedly (code=${code}, signal=${signal})`));
    }
    pendingRpc.clear();

    if (!isQuitting) {
      broadcastEvent({
        event: 'bridge.disconnected',
        data: { code, signal }
      });
      // Auto-restart sidecar after brief delay if app is still open
      setTimeout(() => {
        if (!isQuitting) startPythonBridge();
      }, 1000);
    }
  });
}

function broadcastEvent(payload) {
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) {
      win.webContents.send('autosorter:event', payload);
    }
  }
}

function callBridge(method, params = {}) {
  if (!bridgeProc) {
    startPythonBridge();
  }
  return new Promise((resolve, reject) => {
    if (!bridgeProc || !bridgeProc.stdin.writable) {
      reject(new Error('Python engine is not available. Ensure Python 3.11+ with opencv-python, numpy, and pillow is installed.'));
      return;
    }
    const id = nextRpcId++;
    pendingRpc.set(id, { resolve, reject });
    const line = JSON.stringify({ id, method, params }) + '\n';
    bridgeProc.stdin.write(line, 'utf8', (err) => {
      if (err) {
        pendingRpc.delete(id);
        reject(err);
      }
    });
  });
}

async function stopPythonBridge() {
  if (!bridgeProc) return;
  const proc = bridgeProc;
  try {
    await Promise.race([
      callBridge('bridge.shutdown', {}),
      new Promise((r) => setTimeout(r, 1200))
    ]);
  } catch (_err) {
    // ignore shutdown race
  }
  if (!proc.killed) {
    try {
      proc.kill();
    } catch (_err) {
      // ignore
    }
  }
}

// ---------------------------------------------------------------- Media Protocol

function registerMediaProtocol() {
  protocol.handle('autosorter-media', async (request) => {
    try {
      const parsed = new URL(request.url);
      const rawPath = parsed.searchParams.get('path');
      if (!rawPath) {
        return new Response('Missing path', { status: 400 });
      }

      const resolved = path.resolve(rawPath);
      const ext = path.extname(resolved).toLowerCase();

      if (!NATIVE_IMAGE_EXTS.has(ext) && !CONVERTED_IMAGE_EXTS.has(ext)) {
        return new Response('Unsupported media extension', { status: 403 });
      }

      if (!fs.existsSync(resolved)) {
        return new Response('File not found', { status: 404 });
      }

      let servePath = resolved;
      if (CONVERTED_IMAGE_EXTS.has(ext)) {
        const preview = await callBridge('image.preview', {
          file_path: resolved,
          max_dim: 2200
        });
        servePath = preview.preview_path;
      }

      return net.fetch(pathToFileURL(servePath).toString());
    } catch (err) {
      return new Response(String(err && err.message ? err.message : err), { status: 500 });
    }
  });
}

// ---------------------------------------------------------------- IPC Handlers

function registerIpcHandlers() {
  ipcMain.handle('autosorter:rpc', async (_event, method, params) => {
    return callBridge(method, params);
  });

  ipcMain.handle('autosorter:pickFolder', async (event, options = {}) => {
    const win = BrowserWindow.fromWebContents(event.sender) || mainWindow;
    const result = await dialog.showOpenDialog(win, {
      title: options.title || 'Select folder',
      defaultPath: options.defaultPath || undefined,
      properties: ['openDirectory', 'createDirectory']
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return null;
    }
    return result.filePaths[0];
  });

  ipcMain.handle('autosorter:resolveDroppedPath', async (_event, targetPath) => {
    if (!targetPath) return null;
    try {
      const normalized = path.resolve(targetPath);
      const stat = fs.statSync(normalized);
      if (stat.isDirectory()) return normalized;
      if (stat.isFile()) return path.dirname(normalized);
      return null;
    } catch (_err) {
      return null;
    }
  });

  ipcMain.handle('autosorter:revealInExplorer', async (_event, targetPath) => {
    if (!targetPath) return false;
    const normalized = path.resolve(targetPath);
    if (!fs.existsSync(normalized)) return false;
    shell.showItemInFolder(normalized);
    return true;
  });

  ipcMain.handle('autosorter:openFolder', async (_event, folderPath) => {
    if (!folderPath) return false;
    const normalized = path.resolve(folderPath);
    if (!fs.existsSync(normalized)) return false;
    const errMsg = await shell.openPath(normalized);
    return !errMsg;
  });
}

// ---------------------------------------------------------------- Window

function resolveIconPath() {
  if (app.isPackaged) {
    const packagedIco = path.join(process.resourcesPath, 'icon.ico');
    if (fs.existsSync(packagedIco)) return packagedIco;
  }
  const emeraldIco = path.join(__dirname, 'packaging', 'icon.ico');
  if (fs.existsSync(emeraldIco)) return emeraldIco;
  const ico = path.join(REPO_ROOT, 'packaging', 'app.ico');
  if (fs.existsSync(ico)) return ico;
  const png = path.join(REPO_ROOT, 'autosorter', 'app.png');
  if (fs.existsSync(png)) return png;
  return undefined;
}

function createMainWindow() {
  const isWin = process.platform === 'win32';

  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 920,
    minHeight: 600,
    show: !IS_SELFTEST,
    backgroundColor: '#ffffff',
    icon: resolveIconPath(),
    title: 'Auto-Sorter',
    ...(isWin
      ? {
          titleBarStyle: 'hidden',
          titleBarOverlay: {
            color: '#ffffff',
            symbolColor: '#101b16',
            height: 58
          }
        }
      : {}),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.setMenuBarVisibility(false);
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  if (IS_DEV && !IS_SELFTEST) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  if (IS_SELFTEST) {
    runSelftest(mainWindow);
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function runSelftest(win) {
  const outArg = process.argv.find((a) => a.startsWith('--selftest-out='));
  const outFile = outArg ? outArg.slice('--selftest-out='.length) : null;
  const errors = [];
  win.webContents.on('console-message', (_event, level, message) => {
    if (level >= 3) {
      errors.push(message);
    }
  });
  win.webContents.on('page-error', (err) => {
    errors.push(String(err));
  });

  try {
    await new Promise((resolve) => win.webContents.once('did-finish-load', resolve));
    const sys = await callBridge('system.info', {});
    await new Promise((r) => setTimeout(r, 500));
    const uiCheck = await win.webContents.executeJavaScript(`
      ({
        hasBrand: Boolean(document.querySelector('.brand')),
        hasView: Boolean(document.getElementById('view')),
        viewLength: (document.getElementById('view')?.innerHTML || '').length
      })
    `);
    if (!sys || !sys.models || !sys.models.yunet.present || !sys.models.sface.present) {
      throw new Error('Model verification failed in selftest');
    }
    if (!uiCheck.hasBrand || !uiCheck.hasView || uiCheck.viewLength < 50) {
      throw new Error('Renderer UI failed to mount');
    }
    if (errors.length > 0) {
      throw new Error(`Console/page errors: ${errors.join('; ')}`);
    }
    const payload = { ok: true, packaged: app.isPackaged, sys, uiCheck };
    if (outFile) {
      fs.writeFileSync(outFile, JSON.stringify(payload, null, 2), 'utf8');
    }
    process.stdout.write(`SELFTEST_PASSED ${JSON.stringify(payload)}\n`);
    isQuitting = true;
    await stopPythonBridge();
    app.exit(0);
  } catch (err) {
    const errText = String(err && err.stack ? err.stack : err);
    if (outFile) {
      fs.writeFileSync(outFile, JSON.stringify({ ok: false, error: errText }, null, 2), 'utf8');
    }
    process.stderr.write(`SELFTEST_FAILED: ${errText}\n`);
    isQuitting = true;
    await stopPythonBridge();
    app.exit(1);
  }
}

// ---------------------------------------------------------------- App Lifecycle

if (process.platform === 'win32') {
  app.setAppUserModelId('com.autosorter.desktop');
}

const gotSingleInstanceLock = IS_SELFTEST ? true : app.requestSingleInstanceLock();

if (!gotSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    startPythonBridge();
    registerMediaProtocol();
    registerIpcHandlers();
    createMainWindow();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createMainWindow();
      }
    });
  });
}

app.on('before-quit', async (event) => {
  if (!isQuitting && bridgeProc) {
    event.preventDefault();
    isQuitting = true;
    await stopPythonBridge();
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
