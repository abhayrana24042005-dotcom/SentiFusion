import React from 'react';

const TARGET_SENTIMENTS = [
  { label: 'Positive', icon: '😊', desc: 'Optimistic, favorable' },
  { label: 'Negative', icon: '😞', desc: 'Critical, adverse' },
  { label: 'Neutral', icon: '😐', desc: 'Objective, unbiased' },
  { label: 'Happy', icon: '🎉', desc: 'Joyful, cheerful' },
  { label: 'Sad', icon: '😢', desc: 'Melancholy, sorrow' },
  { label: 'Angry', icon: '😠', desc: 'Frustrated, outraged' },
  { label: 'Motivational', icon: '🚀', desc: 'Inspiring, ambitious' },
  { label: 'Fearful', icon: '😨', desc: 'Apprehensive, alarming' },
  { label: 'Hopeful', icon: '🌟', desc: 'Uplifting, promising' },
  { label: 'Calm', icon: '🌿', desc: 'Serene, peaceful' }
];

export default function TextInput({
  value,
  onChange,
  selectedSentiment,
  onSelectSentiment,
  disabled,
  error
}) {
  const sampleSentences = [
    "This was a triumph, huge success and celebration!",
    "Standard technical manual illustration for routine maintenance.",
    "Critical warning: Severe defect observed during safety inspection."
  ];

  const handleClear = () => {
    onChange('');
  };

  return (
    <div className={`input-card ${error ? 'has-error' : ''}`}>
      <div className="card-header">
        <div className="card-title-group">
          <span className="modality-tag text-modality">Modality 1</span>
          <h2 className="card-title">Target Sentiment & Context</h2>
        </div>
        <span className="char-count">{value.length} chars</span>
      </div>

      <p className="card-hint">
        Select target sentiment to compare against image understanding, or provide text context:
      </p>

      {/* Target Sentiment Quick Select Chips */}
      <div className="sentiment-target-selector">
        <span className="target-select-label">Target Sentiment to Match:</span>
        <div className="target-chips-container">
          {TARGET_SENTIMENTS.map((item) => (
            <button
              key={item.label}
              type="button"
              className={`target-sentiment-chip ${selectedSentiment === item.label ? 'active' : ''}`}
              onClick={() => onSelectSentiment(item.label)}
              disabled={disabled}
              title={item.desc}
            >
              <span className="chip-icon">{item.icon}</span>
              <span className="chip-name">{item.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="textarea-wrapper">
        <textarea
          id="text-input"
          className="text-input-field"
          rows={3}
          placeholder="Optional: Enter context, caption, or user thought to compare..."
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          aria-label="Text input for multimodal sentiment analysis"
        />
        {value.length > 0 && !disabled && (
          <button
            type="button"
            className="clear-text-btn"
            onClick={handleClear}
            title="Clear text"
          >
            ✕
          </button>
        )}
      </div>

      {error && <div className="field-error-message">{error}</div>}

      <div className="sample-chips">
        <span className="sample-label">Quick test context:</span>
        <div className="chip-list">
          {sampleSentences.map((sample, idx) => (
            <button
              key={idx}
              type="button"
              className="sample-chip"
              onClick={() => onChange(sample)}
              disabled={disabled}
            >
              "{sample.slice(0, 32)}..."
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
