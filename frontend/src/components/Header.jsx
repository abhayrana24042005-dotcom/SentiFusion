import React from 'react';
import ThemeToggle from './ThemeToggle';

export default function Header({ theme, onToggleTheme }) {
  return (
    <header className="header-container">
      <div className="header-top-bar">
        <div className="header-title-wrapper">
          <div className="header-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
              <circle cx="12" cy="12" r="4"/>
            </svg>
          </div>
          <div>
            <h1 className="header-title">SentiFusion</h1>
            <p className="header-subtitle">Multimodal Sentiment Analysis</p>
          </div>
        </div>

        {/* Theme Toggle Button */}
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>

      <p className="header-description">
        Dual-modality sentiment analysis architecture combining natural language understanding and visual feature extraction through an attention-based multimodal fusion pipeline.
      </p>
    </header>
  );
}
