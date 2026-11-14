'use strict';

/**
 * Automated Windows Installer (.exe) Build Pipeline for Auto-Sorter Desktop.
 *
 * Stages:
 *   1. Ensure multi-resolution emerald application icon (`electron/packaging/icon.ico`).
 *   2. Freeze Python engine + OpenCV + NumPy + Pillow + ONNX models via PyInstaller
 *      into `build/bridge-dist/autosorter-engine/autosorter-engine.exe`.
 *   3. Smoke-test the frozen `autosorter-engine.exe` binary over stdio (zero system Python).
 *   4. Package Electron app + bundled engine into an installable NSIS `.exe` via `electron-builder`.
 *   5. Run `--selftest` against the packaged `Auto-Sorter.exe` binary to verify end-to-end integrity.
 */

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const ICON_SCRIPT = path.join(__dirname, 'generate_emerald_icon.py');
const ICON_ICO = path.join(__dirname, 'icon.ico');
const SPEC_FILE = path.join(__dirname, 'autosorter_engine.spec');
const BRIDGE_DIST = path.join(REPO_ROOT, 'build', 'bridge-dist');
const BRIDGE_WORK = path.join(REPO_ROOT, 'build', 'bridge-work');
const FROZEN_ENGINE = path.join(BRIDGE_DIST, 'autosorter-engine', 'autosorter-engine.exe');
const RELEASE_DIR = path.join(REPO_ROOT, 'release');

function runChecked(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, {
    cwd: REPO_ROOT,
    stdio: 'inherit',
    shell: false,
    ...opts
  });
  if (res.error) throw res.error;
  if (res.status !== 0) {
    throw new Error(`Command failed with exit code ${res.status}: ${cmd} ${args.join(' ')}`);
  }
}

function verifyFrozenEngine() {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(FROZEN_ENGINE)) {
      reject(new Error(`Frozen engine executable missing at ${FROZEN_ENGINE}`));
      return;
    }

    const proc = spawn(FROZEN_ENGINE, [], {
      cwd: path.dirname(FROZEN_ENGINE),
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdoutBuf = '';
    let gotSystemInfo = false;

    const timer = setTimeout(() => {
      proc.kill();
      reject(new Error('Timed out waiting for frozen autosorter-engine.exe response'));
    }, 15000);

    proc.stdout.on('data', (chunk) => {
      stdoutBuf += chunk.toString('utf8');
      const lines = stdoutBuf.split(/\r?\n/);
      stdoutBuf = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.id === 1) {
          if (!msg.ok || !msg.result?.models?.yunet?.present || !msg.result?.models?.sface?.present) {
            clearTimeout(timer);
            proc.kill();
            reject(new Error(`Frozen engine model check failed: ${line}`));
            return;
          }
          gotSystemInfo = true;
          proc.stdin.write(JSON.stringify({ id: 2, method: 'bridge.shutdown', params: {} }) + '\n');
        }
      }
    });

    proc.on('exit', (code) => {
      clearTimeout(timer);
      if (gotSystemInfo && code === 0) {
        resolve();
      } else {
        reject(new Error(`Frozen engine exited abnormally (code=${code}, gotSystemInfo=${gotSystemInfo})`));
      }
    });

    proc.stdin.write(JSON.stringify({ id: 1, method: 'system.info', params: {} }) + '\n');
  });
}

async function main() {
  console.log('============================================================');
  console.log('Auto-Sorter Desktop — Windows .EXE Installer Build Pipeline');
  console.log('============================================================');

  // Step 1: Icon
  if (!fs.existsSync(ICON_ICO)) {
    console.log('\n[1/5] Generating emerald multi-resolution icon...');
    runChecked('python', [ICON_SCRIPT]);
  } else {
    console.log('\n[1/5] Emerald icon verified:', ICON_ICO);
  }

  // Step 2: Freeze Python RPC engine with PyInstaller
  console.log('\n[2/5] Freezing Python engine + ONNX models via PyInstaller...');
  runChecked('python', [
    '-m',
    'PyInstaller',
    '--noconfirm',
    '--clean',
    '--distpath',
    BRIDGE_DIST,
    '--workpath',
    BRIDGE_WORK,
    SPEC_FILE
  ]);

  // Step 3: Verify standalone frozen engine
  console.log('\n[3/5] Verifying standalone autosorter-engine.exe sidecar...');
  await verifyFrozenEngine();
  console.log('  [OK] autosorter-engine.exe verified (YuNet + SFace bundled, 0 external Python needed).');

  // Step 4: Build NSIS Windows Installer (.exe) with electron-builder
  console.log('\n[4/5] Building Windows NSIS Installer (.exe) with electron-builder...');
  const ebCli = path.join(REPO_ROOT, 'node_modules', 'electron-builder', 'cli.js');
  runChecked(process.execPath, [ebCli, '--win', 'nsis', '--x64'], {
    env: {
      ...process.env,
      CSC_IDENTITY_AUTO_DISCOVERY: 'false'
    }
  });

  // Step 5: Smoke-test packaged Auto-Sorter.exe
  console.log('\n[5/5] Running post-package --selftest on compiled Auto-Sorter.exe...');
  const unpackedExe = path.join(RELEASE_DIR, 'win-unpacked', 'Auto-Sorter.exe');
  if (!fs.existsSync(unpackedExe)) {
    throw new Error(`Expected unpacked executable not found at ${unpackedExe}`);
  }
  const selftestReport = path.join(os.tmpdir(), `autosorter-packaged-selftest-${Date.now()}.json`);
  runChecked(unpackedExe, ['--selftest', `--selftest-out=${selftestReport}`]);

  if (!fs.existsSync(selftestReport)) {
    throw new Error('Packaged Auto-Sorter.exe did not write selftest verification report');
  }
  const reportData = JSON.parse(fs.readFileSync(selftestReport, 'utf8'));
  fs.unlinkSync(selftestReport);
  if (!reportData.ok) {
    throw new Error(`Packaged selftest failed: ${JSON.stringify(reportData)}`);
  }

  const installers = fs
    .readdirSync(RELEASE_DIR)
    .filter((f) => f.toLowerCase().endsWith('.exe'))
    .map((f) => path.join(RELEASE_DIR, f));

  console.log('\n============================================================');
  console.log('SUCCESS: Installable Windows .EXE Ready!');
  for (const inst of installers) {
    const mb = (fs.statSync(inst).size / (1024 * 1024)).toFixed(1);
    console.log(`  Installer: ${inst} (${mb} MB)`);
  }
  console.log('============================================================');
}

main().catch((err) => {
  console.error('\nBUILD FAILED:', err.message || err);
  process.exit(1);
});
