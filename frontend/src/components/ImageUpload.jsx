import React, { useRef, useState } from 'react';
import ImagePreview from './ImagePreview';

const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export default function ImageUpload({ file, previewUrl, onSelectFile, onRemoveFile, disabled, error }) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [fileError, setFileError] = useState(null);

  const validateAndSelect = (selectedFile) => {
    setFileError(null);
    if (!selectedFile) return;

    const fileExt = '.' + selectedFile.name.split('.').pop().toLowerCase();
    const isMimeValid = ALLOWED_MIME_TYPES.includes(selectedFile.type);
    const isExtValid = ALLOWED_EXTENSIONS.includes(fileExt);

    if (!isMimeValid && !isExtValid) {
      setFileError('Invalid file type. Please upload a JPG, JPEG, PNG, or WEBP image.');
      return;
    }

    // 10MB sanity check
    if (selectedFile.size > 10 * 1024 * 1024) {
      setFileError('File size exceeds 10MB limit.');
      return;
    }

    onSelectFile(selectedFile);
  };

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) {
      validateAndSelect(selected);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;

    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      validateAndSelect(droppedFile);
    }
  };

  const handleClickUpload = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.value = null;
      fileInputRef.current.click();
    }
  };

  return (
    <div className={`input-card ${error || fileError ? 'has-error' : ''}`}>
      <div className="card-header">
        <div className="card-title-group">
          <span className="modality-tag image-modality">Modality 2</span>
          <h2 className="card-title">Image Upload</h2>
        </div>
        <span className="format-pills">JPG • PNG • WEBP</span>
      </div>

      <p className="card-hint">
        Select or drag a visual sample for multimodal sentiment analysis:
      </p>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        id="image-file-input"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        style={{ display: 'none' }}
        onChange={handleFileChange}
        disabled={disabled}
      />

      {file && previewUrl ? (
        <ImagePreview
          file={file}
          previewUrl={previewUrl}
          onReplace={handleClickUpload}
          onRemove={onRemoveFile}
          disabled={disabled}
        />
      ) : (
        <div
          className={`dropzone ${isDragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
          onClick={handleClickUpload}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          id="image-dropzone"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') handleClickUpload();
          }}
        >
          <div className="dropzone-icon">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
              <rect x="3" y="3" width="18" height="18" rx="3" ry="3"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
          </div>
          <p className="dropzone-title">Click to upload or drag & drop</p>
          <p className="dropzone-subtitle">Supported formats: JPG, JPEG, PNG, WEBP (up to 10MB)</p>
          <button type="button" className="btn-secondary dropzone-btn" disabled={disabled}>
            Browse Computer
          </button>
        </div>
      )}

      {(fileError || error) && (
        <div className="field-error-message">{fileError || error}</div>
      )}
    </div>
  );
}
