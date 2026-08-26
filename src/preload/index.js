import { contextBridge, webUtils, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  getPathForFile: (file) => webUtils.getPathForFile(file),
  update: {
    check: () => ipcRenderer.invoke('app-update-check'),
    download: () => ipcRenderer.invoke('app-update-download'),
    restart: () => ipcRenderer.send('app-update-restart'),
    onStatus: (callback) => {
      const listener = (_event, payload) => callback(payload)
      ipcRenderer.on('app-update-status', listener)
      return () => ipcRenderer.removeListener('app-update-status', listener)
    }
  }
}

// Disable context menu globally
if (typeof window !== 'undefined') {
  window.addEventListener('contextmenu', (e) => e.preventDefault())
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  window.electron = electronAPI
  window.api = api
}
