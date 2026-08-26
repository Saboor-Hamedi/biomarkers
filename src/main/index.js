import { app, shell, BrowserWindow, ipcMain, Menu } from 'electron'
import { join, basename } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { spawn } from 'child_process'
import fs from 'fs'
import { cpSync, existsSync } from 'fs'
import { autoUpdater } from 'electron-updater'

let pyProcess = null
let serverReady = false
let mainWindow = null

// Production: the bundled `server/` ships unpacked in resources/server.
// Python cannot open files inside app.asar, and the models dir must be writable,
// so we seed a writable runtime copy under userData on first launch.
function getServerDir() {
  if (is.dev) return join(app.getAppPath(), 'server')

  const bundledDir = join(process.resourcesPath, 'server')
  const runtimeDir = join(app.getPath('userData'), 'server')
  if (!existsSync(runtimeDir)) {
    cpSync(bundledDir, runtimeDir, { recursive: true })
  }
  return runtimeDir
}

function waitForServer(proc, timeoutMs = 120000) {
  return new Promise((resolve) => {
    const started = Date.now()
    let done = false
    const finish = (ok) => {
      if (done) return
      done = true
      clearInterval(interval)
      resolve(ok)
    }
    const interval = setInterval(async () => {
      try {
        const res = await fetch('http://127.0.0.1:8001/audit')
        if (res.ok) {
          serverReady = true
          console.log('Biomarker server is ready.')
          finish(true)
          return
        }
      } catch (err) {
        // Not ready yet — keep polling
      }
      if (Date.now() - started > timeoutMs) {
        console.error('Biomarker server did not become ready in time.')
        finish(false)
      }
    }, 1000)
    proc.on('exit', () => finish(false))
    proc.on('error', () => finish(false))
  })
}

async function startPythonServer() {
  try {
    // Attempt to gracefully shut down any existing zombie process
    await fetch('http://127.0.0.1:8001/shutdown', { method: 'POST' }).catch(() => {})
    // Give it a brief moment to shut down
    await new Promise(resolve => setTimeout(resolve, 500))
  } catch (err) {
    // Ignore errors if server wasn't running
  }

  const script = join(getServerDir(), 'main.py')
  const pythonCandidates = process.platform === 'win32' ? ['python', 'py'] : ['python3', 'python']

  for (const pyCmd of pythonCandidates) {
    let spawnError = null
    const proc = spawn(pyCmd, [script], { windowsHide: true })
    proc.on('error', (err) => {
      spawnError = err
      console.error(`Failed to start ${pyCmd}: ${err.message}`)
    })
    proc.stdout.on('data', (data) => console.log(`Python: ${data}`))
    proc.stderr.on('data', (data) => console.error(`Python Error: ${data}`))
    proc.on('exit', (code, signal) => {
      if (pyProcess === proc) pyProcess = null
      console.log(`Python exited: code=${code} signal=${signal}`)
    })

    // Wait until the server actually responds so the UI never sees "failed to fetch"
    const ready = await waitForServer(proc)
    if (ready) {
      pyProcess = proc
      console.log(`Biomarker server started with ${pyCmd}`)
      return
    }

    // Command not found → try the next interpreter. Otherwise it ran but never
    // came up (missing deps, port conflict) → don't retry the same thing.
    if (spawnError) {
      console.error(`Could not launch ${pyCmd}: ${spawnError.message}`)
      continue
    }
    if (proc.exitCode === null) proc.kill()
    break
  }

  console.error('Biomarker server could not be started (is Python + dependencies installed?).')
}

// ── Auto-update ──────────────────────────────────────────────────────────────
function sendUpdateStatus(payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('app-update-status', payload)
  }
}

function setupAutoUpdater() {
  autoUpdater.autoDownload = false
  autoUpdater.autoInstallOnAppQuit = true

  autoUpdater.on('checking-for-update', () => sendUpdateStatus({ type: 'checking' }))
  autoUpdater.on('update-available', (info) =>
    sendUpdateStatus({ type: 'available', version: info.version })
  )
  autoUpdater.on('update-not-available', () => sendUpdateStatus({ type: 'not-available' }))
  autoUpdater.on('error', (err) =>
    sendUpdateStatus({ type: 'error', message: err ? err.message : 'Unknown error' })
  )
  autoUpdater.on('download-progress', (progress) =>
    sendUpdateStatus({
      type: 'progress',
      percent: Math.round(progress.percent),
      transferred: progress.transferred,
      total: progress.total,
      bytesPerSecond: progress.bytesPerSecond
    })
  )
  autoUpdater.on('update-downloaded', (info) =>
    sendUpdateStatus({ type: 'downloaded', version: info.version })
  )
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    title: 'Cancer Biomarker AI Suite',
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // Enable DevTools toggle with Ctrl+Shift+I (menu is removed, so bind manually)
  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (
      input.type === 'keyDown' &&
      input.control &&
      input.shift &&
      input.key.toLowerCase() === 'i'
    ) {
      event.preventDefault()
      if (mainWindow.webContents.isDevToolsOpened()) {
        mainWindow.webContents.closeDevTools()
      } else {
        mainWindow.webContents.openDevTools({ mode: 'detach' })
      }
    }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.electron')
  Menu.setApplicationMenu(null)
  app.on('browser-window-created', (_, window) => {
    window.autoHideMenuBar = true
    window.setMenuBarVisibility(false)
    optimizer.watchWindowShortcuts(window)
  })

  // IPC: Synchronize Artifacts — copies .pkl files into the writable models dir
  ipcMain.handle('sync-artifacts', async (event, filePaths) => {
    const destDir = join(getServerDir(), 'analysis', 'models')
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true })
    }

    const results = []
    for (const filePath of filePaths) {
      try {
        const destPath = join(destDir, basename(filePath))
        fs.copyFileSync(filePath, destPath)
        results.push({ name: basename(filePath), status: 'success' })
      } catch (err) {
        results.push({ name: basename(filePath), status: 'error', error: err.message })
      }
    }
    return results
  })

  // IPC: Reset Artifacts — clears UI state only, NEVER deletes model files
  ipcMain.handle('reset-artifacts', async () => {
    // The user explicitly requested to never delete the .pkl models.
    // This handler now solely exists to let the frontend wipe its UI state without touching the filesystem.
    return { status: 'success' }
  })

  ipcMain.handle('check-audit-status', async () => {
    try {
      const response = await fetch('http://127.0.0.1:8001/audit')
      if (!response.ok) return { error: 'Server Offline' }
      return await response.json()
    } catch (err) {
      return { error: 'Failed to fetch' }
    }
  })

  ipcMain.handle('check-top-patients', async () => {
    try {
      const response = await fetch('http://127.0.0.1:8001/top-patients')
      if (!response.ok) return { error: 'Server Offline' }
      return await response.json()
    } catch (err) {
      return { error: 'Failed to fetch' }
    }
  })

  ipcMain.on('ping', () => console.log('pong'))

  // ── Auto-update IPC ─────────────────────────────────────────────────────────
  ipcMain.handle('app-update-check', async () => {
    try {
      await autoUpdater.checkForUpdates()
      return { status: 'ok' }
    } catch (err) {
      return { status: 'error', message: err.message }
    }
  })

  ipcMain.handle('app-update-download', async () => {
    try {
      await autoUpdater.downloadUpdate()
      return { status: 'ok' }
    } catch (err) {
      return { status: 'error', message: err.message }
    }
  })

  ipcMain.on('app-update-restart', () => {
    // Make sure the Python server is stopped before the installer replaces the app
    if (pyProcess) {
      pyProcess.kill()
      pyProcess = null
    }
    setTimeout(() => autoUpdater.quitAndInstall(), 300)
  })

  setupAutoUpdater()
  createWindow()
  startPythonServer()

  // Silently check for updates in the background (packaged builds only)
  if (app.isPackaged) {
    autoUpdater.checkForUpdates()
  }

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (pyProcess) {
    pyProcess.kill()
    pyProcess = null
  }
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
