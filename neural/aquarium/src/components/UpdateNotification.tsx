import React, { useEffect, useState } from 'react';
import './UpdateNotification.css';

interface UpdateInfo {
  version: string;
  releaseNotes?: string;
  releaseDate?: string;
}

interface UpdateProgress {
  bytesPerSecond: number;
  percent: number;
  transferred: number;
  total: number;
}

const UpdateNotification: React.FC = () => {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [updateDownloaded, setUpdateDownloaded] = useState(false);
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<UpdateProgress | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!window.electron) {
      return;
    }

    window.electron.onUpdateAvailable((info: UpdateInfo) => {
      setUpdateAvailable(true);
      setUpdateInfo(info);
      setIsDownloading(true);
    });

    window.electron.onUpdateDownloadProgress((progress: UpdateProgress) => {
      setDownloadProgress(progress);
    });

    window.electron.onUpdateDownloaded((info: UpdateInfo) => {
      setUpdateDownloaded(true);
      setIsDownloading(false);
      setUpdateInfo(info);
    });

    window.electron.onUpdateError((err: Error) => {
      setError(err.message || 'Update failed');
      setIsDownloading(false);
    });

    return () => {
      if (window.electron) {
        window.electron.removeAllListeners?.('update-available');
        window.electron.removeAllListeners?.('update-download-progress');
        window.electron.removeAllListeners?.('update-downloaded');
        window.electron.removeAllListeners?.('update-error');
      }
    };
  }, []);

  const handleCheckForUpdates = async () => {
    if (window.electron) {
      setError(null);
      await window.electron.checkForUpdates();
    }
  };

  const handleInstallUpdate = async () => {
    if (window.electron) {
      await window.electron.installUpdate();
    }
  };

  const handleDismiss = () => {
    setUpdateAvailable(false);
    setUpdateDownloaded(false);
    setError(null);
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const formatSpeed = (bytesPerSecond: number): string => {
    return formatBytes(bytesPerSecond) + '/s';
  };

  if (error) {
    return (
      <div className="update-notification update-error">
        <div className="update-icon">⚠️</div>
        <div className="update-content">
          <div className="update-title">Update Error</div>
          <div className="update-message">{error}</div>
        </div>
        <button className="update-dismiss" onClick={handleDismiss}>
          ✕
        </button>
      </div>
    );
  }

  if (updateDownloaded) {
    return (
      <div className="update-notification update-ready">
        <div className="update-icon">✓</div>
        <div className="update-content">
          <div className="update-title">Update Ready</div>
          <div className="update-message">
            Version {updateInfo?.version} has been downloaded and is ready to install.
          </div>
        </div>
        <div className="update-actions">
          <button className="update-button update-install" onClick={handleInstallUpdate}>
            Restart & Install
          </button>
          <button className="update-button update-later" onClick={handleDismiss}>
            Later
          </button>
        </div>
      </div>
    );
  }

  if (isDownloading && downloadProgress) {
    return (
      <div className="update-notification update-downloading">
        <div className="update-icon">⬇️</div>
        <div className="update-content">
          <div className="update-title">Downloading Update</div>
          <div className="update-message">
            Version {updateInfo?.version} - {Math.round(downloadProgress.percent)}%
          </div>
          <div className="update-progress-bar">
            <div
              className="update-progress-fill"
              style={{ width: `${downloadProgress.percent}%` }}
            />
          </div>
          <div className="update-stats">
            {formatBytes(downloadProgress.transferred)} / {formatBytes(downloadProgress.total)}
            {' • '}
            {formatSpeed(downloadProgress.bytesPerSecond)}
          </div>
        </div>
      </div>
    );
  }

  if (updateAvailable && !isDownloading) {
    return (
      <div className="update-notification update-available">
        <div className="update-icon">🔔</div>
        <div className="update-content">
          <div className="update-title">Update Available</div>
          <div className="update-message">
            Version {updateInfo?.version} is available. Downloading in background...
          </div>
        </div>
        <button className="update-dismiss" onClick={handleDismiss}>
          ✕
        </button>
      </div>
    );
  }

  return null;
};

export default UpdateNotification;

declare global {
  interface Window {
    electron?: {
      checkForUpdates: () => Promise<void>;
      installUpdate: () => Promise<void>;
      getAppVersion: () => Promise<string>;
      onUpdateAvailable: (callback: (info: UpdateInfo) => void) => void;
      onUpdateNotAvailable: (callback: (info: UpdateInfo) => void) => void;
      onUpdateDownloadProgress: (callback: (progress: UpdateProgress) => void) => void;
      onUpdateDownloaded: (callback: (info: UpdateInfo) => void) => void;
      onUpdateError: (callback: (error: Error) => void) => void;
      onMenuNewProject: (callback: () => void) => void;
      onMenuOpenProject: (callback: () => void) => void;
      onMenuSave: (callback: () => void) => void;
      removeAllListeners?: (channel: string) => void;
    };
  }
}
