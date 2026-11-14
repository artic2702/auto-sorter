'use strict';

const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('autosorter', {
  platform: process.platform,

  /** Invoke any Python RPC method (e.g. 'library.inspect', 'people.list'). */
  rpc(method, params = {}) {
    return ipcRenderer.invoke('autosorter:rpc', method, params);
  },

  /** Open native OS folder picker dialog. */
  pickFolder(options = {}) {
    return ipcRenderer.invoke('autosorter:pickFolder', options);
  },

  /** Resolve a dropped File/Directory path safely using Electron's webUtils. */
  resolveDroppedPath(file) {
    try {
      const p = webUtils.getPathForFile(file);
      return ipcRenderer.invoke('autosorter:resolveDroppedPath', p);
    } catch (_err) {
      return Promise.resolve(null);
    }
  },

  /** Reveal a file or folder in Windows File Explorer. */
  revealInExplorer(targetPath) {
    return ipcRenderer.invoke('autosorter:revealInExplorer', targetPath);
  },

  /** Open a folder directly in Windows File Explorer. */
  openFolder(folderPath) {
    return ipcRenderer.invoke('autosorter:openFolder', folderPath);
  },

  /** Build a secure `autosorter-media://` URL for thumbnails or full-res photos. */
  mediaUrl(filePath, cacheBust = '') {
    if (!filePath) return '';
    const q = new URLSearchParams({ path: String(filePath) });
    if (cacheBust) q.set('v', String(cacheBust));
    return `autosorter-media://local/?${q.toString()}`;
  },

  /** Subscribe to async events pushed from the Python sidecar ('pipeline.progress', etc.). */
  onEvent(callback) {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('autosorter:event', listener);
    return () => ipcRenderer.removeListener('autosorter:event', listener);
  }
});
