import React from 'react';

export default function AnalyzeButton({ onClick, isLoading, disabled }) {
  return (
    <div className="analyze-action-container">
      <button
        type="button"
        id="analyze-btn"
        className={`analyze-button ${isLoading ? 'loading' : ''}`}
        onClick={onClick}
        disabled={disabled || isLoading}
        aria-busy={isLoading}
      >
        {isLoading ? (
          <span className="btn-content">
            <svg className="spinner-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
            </svg>
            Analyzing...
          </span>
        ) : (
          <span className="btn-content">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
            Analyze
          </span>
        )}
      </button>
    </div>
  );
}
