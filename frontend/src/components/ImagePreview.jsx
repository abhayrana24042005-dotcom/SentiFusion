import React from 'react';

export default function ImagePreview({ file, previewUrl, onReplace, onRemove, disabled }) {
  if (!file || !previewUrl) return null;

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="preview-container">
      <div className="preview-image-wrapper">
        <img
          src={previewUrl}
          alt="Selected preview"
          className="preview-image"
          id="uploaded-image-preview"
        />
        <div className="preview-overlay">
          <span className="preview-badge">Ready for Pipeline</span>
        </div>
      </div>

      <div className="preview-meta">
        <div className="file-info">
          <p className="file-name" title={file.name}>{file.name}</p>
          <p className="file-size">{formatFileSize(file.size)} • {file.type || 'image'}</p>
        </div>

        <div className="preview-actions">
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={onReplace}
            disabled={disabled}
            id="replace-image-btn"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
              <path d="M3 3v5h5"/>
              <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
              <path d="M16 21h5v-5"/>
            </svg>
            Replace
          </button>
          <button
            type="button"
            className="btn-danger btn-sm"
            onClick={onRemove}
            disabled={disabled}
            id="remove-image-btn"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
            Remove
          </button>
        </div>
      </div>
    </div>
  );
}
