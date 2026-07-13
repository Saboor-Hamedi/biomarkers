# Electron Desktop App

The Electron shell wraps the React frontend, manages the Python backend process, and handles file system operations.

## Source File

`src/main/index.js`

## Responsibilities

### 1. Python Process Management

On app ready, spawns the FastAPI server as a child process:

```javascript
let pythonProcess = spawn('python', ['-u', 'server/main.py'], {
  env: { ...process.env, PYTHONUNBUFFERED: '1' }
});
```

Key behaviors:
- Spawns on port **8001** (canonical server port)
- Killed on app quit via `pythonProcess.kill()`
- Periodically sends keep-alive heartbeats (`POST /reset` every 60s)
- Logs Python stdout/stderr to console

### 2. IPC Handlers

| Channel | Direction | Purpose |
|---------|-----------|---------|
| `predict` | Renderer → Main | Forward `/predict` POST to localhost:8001 |
| `stats` | Renderer → Main | Forward `/stats` GET |
| `ensemble` | Renderer → Main | Forward `/analyze-batch` POST |
| `metrics` | Renderer → Main | Forward `/metrics` GET |
| `sample-ids` | Renderer → Main | Forward `/sample-ids` GET |
| `concentrations` | Renderer → Main | Forward `/concentrations/{id}` GET |
| `concentrations-batch` | Renderer → Main | Forward `/concentrations/batch` POST |
| `reset-artifacts` | Renderer → Main | **No-op** (does not delete models) |
| `sync-artifacts` | Renderer → Main | Synced files from `server/analysis/models/` to `server/analysis/models/` (identity copy) |
| `confirm-dialog` | Renderer → Main | Standard dialog.showMessageBox wrapper |

**IPC forwarding pattern** (HTTP bridge):
```javascript
ipcMain.handle('predict', async (_, data) => {
  const response = await fetch('http://localhost:8001/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return response.json();
});
```

### 3. Application Window

- Created via `electron-window-state` for persisted position/size
- Loads Vite dev server in dev mode, production build otherwise
- Context menu disabled globally

### 4. Process Lifecycle

```
App ready
  ↓
Create BrowserWindow
  ↓
Spawn Python server (port 8001)
  ↓
Wait for Python ready (poll GET /)
  ↓
Load frontend
  ↓
[User interacts → IPC ←→ HTTP :8001]
  ↓
App quit
  ↓
Kill Python process
```

## Preload Script

`src/preload/index.js`

Exposes `electronAPI` (from `@electron-toolkit/preload`) and `api.getPathForFile()` via `contextBridge`.

## Configuration

`electron-builder.yml` — Packaging config:
- App ID: `com.electron.app`
- Product name: configurable
- Directories: `buildResources`, `output`
- File associations and installer config
