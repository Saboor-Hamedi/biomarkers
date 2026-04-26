import { app, shell, BrowserWindow, ipcMain, Menu } from 'electron'
import { join, basename } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { spawn } from 'child_process'
import fs from 'fs'

let pyProcess = null

function startPythonServer() {
  const script = join(__dirname, '../../server/main.py')
  pyProcess = spawn('python', [script])
  pyProcess.stdout.on('data', (data) => console.log(`Python: ${data}`))
  pyProcess.stderr.on('data', (data) => console.error(`Python Error: ${data}`))
}

function createWindow() {
  const mainWindow = new BrowserWindow({
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

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
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

  // IPC: Synchronize Artifacts
  ipcMain.handle('sync-artifacts', async (event, filePaths) => {
    const destDir = join(app.getAppPath(), 'server', 'artifacts')
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

  // IPC: Reset Artifacts
  ipcMain.handle('reset-artifacts', async () => {
    const destDir = join(app.getAppPath(), 'server', 'artifacts')
    if (fs.existsSync(destDir)) {
      const files = fs.readdirSync(destDir)
      for (const file of files) {
        fs.unlinkSync(join(destDir, file))
      }
    }
    return { status: 'success' }
  })

  ipcMain.handle('check-audit-status', async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/audit')
      if (!response.ok) return { error: 'Server Offline' }
      return await response.json()
    } catch (err) {
      return { error: 'Failed to fetch' }
    }
  })

  ipcMain.handle('check-top-patients', async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/top-patients')
      if (!response.ok) return { error: 'Server Offline' }
      return await response.json()
    } catch (err) {
      return { error: 'Failed to fetch' }
    }
  })

  ipcMain.on('ping', () => console.log('pong'))

  startPythonServer()
  createWindow()

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
