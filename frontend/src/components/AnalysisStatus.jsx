import React, { useState } from 'react';

const formatValue = (val) => {
  if (val === null || val === undefined) return '';
  if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') {
    return String(val);
  }
  if (typeof val === 'object') {
    if (val.label) return val.confidence ? `${val.label} (${Math.round(val.confidence * 100)}%)` : val.label;
    if (val.text) return val.text;
    if (val.explanation) return val.explanation;
    if (val.description) return val.description;
    try {
      return JSON.stringify(val);
    } catch {
      return '[Object]';
    }
  }
  return String(val);
};

export default function AnalysisStatus({
  isLoading,
  statusData,
  errorMessage,
  onReset,
  selectedSentiment,
  onSelectSentiment
}) {
  const [feedbackSaved, setFeedbackSaved] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState('');

  if (!isLoading && !statusData && !errorMessage) {
    return (
      <div className="status-idle-placeholder">
        <div className="idle-indicator">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="16" x2="12" y2="12"/>
            <line x1="12" y1="8" x2="12.01" y2="8"/>
          </svg>
          <span>Ready for multimodal analysis. Select a sentiment, upload an image, and click <strong>Analyze Image & Sentiment</strong>.</span>
        </div>
      </div>
    );
  }

  // Handle Feedback Submission
  const handleSendFeedback = async (isCorrect) => {
    if (!statusData || !statusData.image_hash) return;
    try {
      const host = (typeof window !== 'undefined' && window.location.hostname) ? window.location.hostname : '127.0.0.1';
      const formData = new FormData();
      formData.append('image_hash', statusData.image_hash);
      formData.append('user_sentiment', selectedSentiment || 'Neutral');
      formData.append('predicted_sentiment', statusData.prediction?.sentiment || 'Neutral');
      formData.append('match_score', statusData.prediction?.sentiment_match?.match_score || 50);
      formData.append('feedback_label', isCorrect ? 'correct' : 'incorrect');

      const res = await fetch(`http://${host}:8000/feedback`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setFeedbackSaved(true);
      setFeedbackMessage(isCorrect ? '✓ Evaluation recorded as accurate!' : '✓ Correction logged for dataset evaluation.');
    } catch (err) {
      setFeedbackMessage('Failed to save feedback: ' + err.message);
    }
  };

  const prediction = statusData?.prediction;
  const imageUnderstanding = prediction?.image_understanding || {};
  const sentimentMatch = prediction?.sentiment_match || {};
  const perspectives = prediction?.perspectives || imageUnderstanding?.perspectives || {};
  const perf = statusData?.performance || imageUnderstanding?.performance || {};
  const ocrText = Array.isArray(imageUnderstanding?.image?.ocr_text) ? imageUnderstanding.image.ocr_text : [];
  const detectedObjects = Array.isArray(imageUnderstanding?.image?.objects) ? imageUnderstanding.image.objects : [];
  const scene = formatValue(imageUnderstanding?.image?.scene || 'Unspecified');
  const emotions = imageUnderstanding?.emotion_analysis || {};
  const uncertainty = Array.isArray(imageUnderstanding?.uncertainty) ? imageUnderstanding.uncertainty : [];

  const matchScore = sentimentMatch.match_score ?? (prediction?.confidence || 0);
  const matchLevel = formatValue(sentimentMatch.match_level || 'Moderate Match');
  const detectedSentiment = formatValue(sentimentMatch.detected_sentiment || prediction?.sentiment || 'Neutral');
  const isCacheHit = perf.cache_hit === true;

  const getMatchColor = (score) => {
    if (score >= 70) return '#10b981';
    if (score >= 40) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <section className="analysis-status-section" aria-live="polite">
      {/* Loading State */}
      {isLoading && (
        <div className="status-card status-loading" id="status-loading-card">
          <div className="status-header">
            <div className="status-spinner-large"></div>
            <div>
              <h3 className="status-title">Performing Intelligent Multimodal Analysis...</h3>
              <p className="status-sub">
                Extracting visual entities, OCR text, facial expressions, deep context, and matching target sentiment...
              </p>
            </div>
          </div>
          <div className="progress-bar-animated">
            <div className="progress-bar-fill"></div>
          </div>
        </div>
      )}

      {/* Error State */}
      {errorMessage && !isLoading && (
        <div className="status-card status-error" id="status-error-card">
          <div className="status-icon-badge error-badge">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <line x1="15" y1="9" x2="9" y2="15"/>
              <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
          </div>
          <div className="status-content">
            <h3 className="status-title error-title">Analysis / Service Error</h3>
            <p className="status-message" id="error-message-text">{formatValue(errorMessage)}</p>
          </div>
          <button type="button" className="btn-secondary btn-sm" onClick={onReset}>
            Dismiss
          </button>
        </div>
      )}

      {/* Main Results Display */}
      {statusData && prediction && !isLoading && (
        <div className="status-card status-results-card" id="status-success-card">
          
          {/* Cache & Telemetry Banner */}
          <div className="telemetry-banner">
            <div className="telemetry-item">
              <span className={`cache-badge ${isCacheHit ? 'cache-hit' : 'cache-miss'}`}>
                {isCacheHit ? '⚡ Cached image analysis reused (<1ms)' : '✨ New image analyzed'}
              </span>
            </div>
            {statusData.image_hash && (
              <div className="telemetry-item hash-tag">
                <span className="hash-label">SHA-256:</span>
                <code>{String(statusData.image_hash).slice(0, 12)}...</code>
              </div>
            )}
            <div className="telemetry-item latency-tag">
              <span>Latency: <strong>{perf.processing_time_ms || 1} ms</strong></span>
            </div>
          </div>

          {/* Sentiment Match Meter Hero */}
          <div className="match-meter-hero">
            <div className="match-header-row">
              <div className="sentiment-comparison-titles">
                <div className="comp-item user-sent">
                  <span className="comp-label">Target Sentiment:</span>
                  <span className="comp-val user-val">{formatValue(sentimentMatch.user_selected_sentiment || selectedSentiment || 'User Target')}</span>
                </div>
                <div className="comp-vs">vs</div>
                <div className="comp-item detected-sent">
                  <span className="comp-label">Detected Image Sentiment:</span>
                  <span className="comp-val detected-val">{detectedSentiment}</span>
                </div>
              </div>

              <div className="match-score-badge" style={{ borderColor: getMatchColor(matchScore) }}>
                <span className="match-score-number" style={{ color: getMatchColor(matchScore) }}>
                  {matchScore}%
                </span>
                <span className="match-score-sub">{matchLevel}</span>
              </div>
            </div>

            {/* Score Track */}
            <div className="match-track-container">
              <div
                className="match-track-fill"
                style={{
                  width: `${Math.min(100, Math.max(5, matchScore))}%`,
                  backgroundColor: getMatchColor(matchScore)
                }}
              ></div>
            </div>

            <p className="match-explanation">
              <strong>Match Explanation:</strong> {formatValue(sentimentMatch.explanation || 'Analyzed visual context and user sentiment compatibility.')}
            </p>
          </div>

          {/* Quick Sentiment Re-Match Bar */}
          <div className="quick-rematch-bar">
            <span className="rematch-title">⚡ Instant Re-Match (Zero vision re-run):</span>
            <div className="rematch-chips">
              {['Positive', 'Negative', 'Neutral', 'Happy', 'Sad', 'Angry', 'Motivational', 'Fearful', 'Hopeful'].map((sent) => (
                <button
                  key={sent}
                  type="button"
                  className={`rematch-chip ${selectedSentiment === sent ? 'active' : ''}`}
                  onClick={() => onSelectSentiment && onSelectSentiment(sent)}
                >
                  {sent}
                </button>
              ))}
            </div>
          </div>

          {/* Deep Image Understanding & Message Card */}
          <div className="understanding-card">
            <div className="card-section-title">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2a10 10 0 100 20 10 10 0 000-20z"/>
                <path d="M12 16v-4"/>
                <path d="M12 8h.01"/>
              </svg>
              <h3>Deep Visual Perception & Hierarchical Understanding</h3>
            </div>
            <p className="image-description-text">
              {formatValue(imageUnderstanding.image?.description || imageUnderstanding.scene?.summary || 'Natural language image overview.')}
            </p>
            {imageUnderstanding.message?.interpretation && (
              <div className="image-intent-pill">
                <span className="intent-label">Inferred Intent / Story:</span>
                <span className="intent-val">{formatValue(imageUnderstanding.message.interpretation)}</span>
              </div>
            )}
          </div>

          {/* Cross-Modal Synthesis & Sarcasm Resolution Banner */}
          {prediction.cross_modal_analysis && (
            <div className="cross-modal-card" style={{
              background: prediction.cross_modal_conflict?.has_conflict || prediction.cross_modal_conflict?.conflict_detected ? '#fef2f2' : '#f0fdf4',
              border: `1px solid ${prediction.cross_modal_conflict?.has_conflict || prediction.cross_modal_conflict?.conflict_detected ? '#fecaca' : '#bbf7d0'}`,
              borderRadius: '12px',
              padding: '16px',
              margin: '16px 0'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <h4 style={{ margin: 0, fontSize: '1.05rem', color: prediction.cross_modal_conflict?.has_conflict || prediction.cross_modal_conflict?.conflict_detected ? '#991b1b' : '#166534' }}>
                  {prediction.cross_modal_conflict?.has_conflict || prediction.cross_modal_conflict?.conflict_detected ? '⚠️ Cross-Modal Incongruity / Sarcasm Detected' : '🤝 Harmonious Cross-Modal Alignment'}
                </h4>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, padding: '3px 10px', borderRadius: '12px', background: '#fff' }}>
                  {prediction.cross_modal_analysis?.semantic_relationship || 'Evaluated'}
                </span>
              </div>
              <p style={{ margin: '6px 0 10px 0', fontSize: '0.92rem', lineHeight: '1.45', color: '#334155' }}>
                <strong>Combined Interpretation:</strong> {formatValue(prediction.cross_modal_analysis?.combined_interpretation?.interpretation || prediction.final_prediction?.reasoning)}
              </p>
              {prediction.cross_modal_analysis?.modality_importance && (
                <div style={{ fontSize: '0.85rem', color: '#475569', borderTop: '1px dashed #cbd5e1', paddingTop: '8px' }}>
                  <strong>Modality Rationale:</strong> {formatValue(prediction.cross_modal_analysis.modality_importance.rationale || prediction.cross_modal_analysis.modality_importance.importance)}
                </div>
              )}
            </div>
          )}

          {/* Living & Non-Living Entities Perception Grid */}
          {imageUnderstanding.image_analysis && (
            <div className="deep-entities-section" style={{ margin: '18px 0' }}>
              <h4 className="section-label" style={{ marginBottom: '10px' }}>Categorized Visual Entities & Spatial Layout</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
                
                {/* Living Entities */}
                <div className="entity-box" style={{ background: '#f8fafc', padding: '14px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
                  <h5 style={{ margin: '0 0 10px 0', color: '#0f766e', fontSize: '0.95rem' }}>🌿 Living Entities (Biological / Human / Fauna)</h5>
                  {Array.isArray(imageUnderstanding.image_analysis.living_entities) && imageUnderstanding.image_analysis.living_entities.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {imageUnderstanding.image_analysis.living_entities.map((e, idx) => (
                        <div key={idx} style={{ background: '#fff', padding: '8px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.86rem' }}>
                          <span style={{ fontWeight: 600, color: '#0f172a' }}>{e.entity}</span> ({e.subcategory})
                          <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '3px' }}>
                            Location: {e.location} | Size: {e.relative_size} | Conf: {Math.round((e.confidence || 0.8) * 100)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: '0.85rem', color: '#64748b' }}>No living entities detected</div>
                  )}
                </div>

                {/* Non-Living Entities */}
                <div className="entity-box" style={{ background: '#f8fafc', padding: '14px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
                  <h5 style={{ margin: '0 0 10px 0', color: '#1e40af', fontSize: '0.95rem' }}>💻 Non-Living Entities & Objects</h5>
                  {Array.isArray(imageUnderstanding.image_analysis.non_living_entities) && imageUnderstanding.image_analysis.non_living_entities.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {imageUnderstanding.image_analysis.non_living_entities.slice(0, 5).map((e, idx) => (
                        <div key={idx} style={{ background: '#fff', padding: '8px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.86rem' }}>
                          <span style={{ fontWeight: 600, color: '#0f172a' }}>{e.entity}</span> ({e.subcategory})
                          <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '3px' }}>
                            Location: {e.location} | Size: {e.relative_size} | Conf: {Math.round((e.confidence || 0.8) * 100)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: '0.85rem', color: '#64748b' }}>No physical objects isolated</div>
                  )}
                </div>

              </div>
            </div>
          )}

          {/* Entity Relationships & Actions Graph */}
          {imageUnderstanding.image_analysis?.relationships && imageUnderstanding.image_analysis.relationships.length > 0 && (
            <div className="relationships-section" style={{ margin: '18px 0', background: '#f8fafc', padding: '14px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
              <h4 style={{ margin: '0 0 10px 0', fontSize: '0.98rem', color: '#334155' }}>🔗 Entity Relationship Graph (Direct Visual Backing)</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                {imageUnderstanding.image_analysis.relationships.map((rel, idx) => (
                  <div key={idx} style={{ background: '#fff', padding: '10px 14px', borderRadius: '8px', border: '1px solid #cbd5e1', flex: '1 1 240px', fontSize: '0.85rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: '#1e293b' }}>
                      <span>{rel.subject}</span>
                      <span style={{ color: '#6366f1' }}>➔ [{rel.relationship}] ➔</span>
                      <span>{rel.object}</span>
                    </div>
                    <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '4px' }}>
                      Evidence: {rel.evidence}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Implicit Meaning & Evidence Tiers Card */}
          {imageUnderstanding.image_analysis?.implicit_meaning && (
            <div className="implicit-meaning-card" style={{ margin: '18px 0', background: '#f1f5f9', padding: '14px', borderRadius: '10px', border: '1px solid #cbd5e1' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                <h4 style={{ margin: 0, fontSize: '0.98rem', color: '#1e293b' }}>💡 Implicit & Contextual Meaning Analysis</h4>
                <span style={{ fontSize: '0.76rem', fontWeight: 700, padding: '2px 8px', borderRadius: '6px', background: '#e2e8f0', color: '#334155' }}>
                  EVIDENCE TIER: {imageUnderstanding.image_analysis.implicit_meaning.evidence_level || 'INFERRED'}
                </span>
              </div>
              <p style={{ margin: '4px 0 8px 0', fontSize: '0.9rem', color: '#334155' }}>
                {imageUnderstanding.image_analysis.implicit_meaning.meaning}
              </p>
              {Array.isArray(imageUnderstanding.image_analysis.implicit_meaning.supporting_evidence) && (
                <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
                  <strong>Supporting Evidence:</strong> {imageUnderstanding.image_analysis.implicit_meaning.supporting_evidence.join('; ')}
                </div>
              )}
            </div>
          )}

          {/* Visual Entities & OCR Extraction Grid */}
          <div className="visual-entities-grid">
            
            {/* Objects & People */}
            <div className="entity-box">
              <h4 className="entity-box-title">👁️ Visual Detection</h4>
              <div className="entity-detail-list">
                <div className="entity-item">
                  <span className="lbl">People Count:</span>
                  <span className="val">{formatValue(imageUnderstanding.image?.people_count ?? 0)}</span>
                </div>
                <div className="entity-item">
                  <span className="lbl">Facial Expressions:</span>
                  <span className="val">{formatValue(imageUnderstanding.image?.facial_expressions || 'None detected')}</span>
                </div>
                <div className="entity-item">
                  <span className="lbl">Scene / Environment:</span>
                  <span className="val">{scene}</span>
                </div>
                <div className="entity-item">
                  <span className="lbl">Atmosphere:</span>
                  <span className="val">{formatValue(imageUnderstanding.image?.visual_atmosphere || 'Standard')}</span>
                </div>
              </div>

              {detectedObjects.length > 0 && (
                <div className="object-tags-row">
                  <span className="tags-label">Detected Entities:</span>
                  <div className="tags-container">
                    {detectedObjects.map((obj, i) => (
                      <span key={i} className="object-tag">
                        {typeof obj === 'object' && obj.label ? `${obj.label} (${Math.round((obj.confidence || 0) * 100)}%)` : formatValue(obj)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* OCR & Emotion Details */}
            <div className="entity-box">
              <h4 className="entity-box-title">📝 Visible Text (OCR) & Emotions</h4>
              
              {/* OCR Text Box */}
              <div className="ocr-results-box">
                <span className="ocr-heading">Extracted Text:</span>
                {ocrText.length > 0 ? (
                  <div className="ocr-chips-list">
                    {ocrText.map((t, i) => (
                      <span key={i} className="ocr-chip">"{formatValue(t)}"</span>
                    ))}
                  </div>
                ) : (
                  <span className="ocr-empty-note">No text visible in image</span>
                )}
              </div>

              {/* Emotion Details */}
              <div className="emotion-detail-box">
                <div className="emotion-row">
                  <span className="lbl">Primary Emotion:</span>
                  <span className="val emotion-val">{formatValue(emotions.primary || 'Neutral')} ({Math.round((emotions.confidence || 0.5) * 100)}%)</span>
                </div>
                {Array.isArray(emotions.evidence) && emotions.evidence.length > 0 && (
                  <p className="emotion-evidence-text">
                    <strong>Evidence:</strong> {emotions.evidence.map(formatValue).join('; ')}
                  </p>
                )}
              </div>

            </div>

          </div>

          {/* 8-Perspective Comprehensive Analysis Grid */}
          <div className="perspectives-container">
            <h4 className="section-label">Multi-Perspective Analytical Breakdown</h4>
            <div className="perspectives-grid">
              {Object.entries(perspectives).map(([key, val]) => (
                <div key={key} className="perspective-card">
                  <div className="perspective-name">{key.replace(/_/g, ' ')}</div>
                  <div className="perspective-content">{formatValue(val)}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Sarcasm / Incongruity / Uncertainty Warnings */}
          {uncertainty.length > 0 && (
            <div className="uncertainty-card">
              <div className="uncertainty-header">
                <span className="warn-icon">⚠️</span>
                <h4>Uncertainty & Ambiguity Notes</h4>
              </div>
              <ul className="uncertainty-list">
                {uncertainty.map((item, idx) => (
                  <li key={idx}>{formatValue(item)}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Continuous Quality Evaluation / Feedback Section */}
          <div className="feedback-eval-section">
            <div className="feedback-text-info">
              <span className="feedback-title">Rate Analysis Accuracy for System Evaluation:</span>
              <span className="feedback-sub">Logs test cases for precision, recall, and confusion matrix tracking.</span>
            </div>
            
            {!feedbackSaved ? (
              <div className="feedback-buttons-row">
                <button
                  type="button"
                  className="btn-feedback correct"
                  onClick={() => handleSendFeedback(true)}
                >
                  ✓ Accurate Analysis
                </button>
                <button
                  type="button"
                  className="btn-feedback incorrect"
                  onClick={() => handleSendFeedback(false)}
                >
                  ✗ Report Mismatch
                </button>
              </div>
            ) : (
              <div className="feedback-confirmed-msg">{feedbackMessage}</div>
            )}
          </div>

          {/* Footer Reset Action */}
          <div className="status-footer">
            <button type="button" className="btn-secondary btn-sm" onClick={onReset} id="new-analysis-btn">
              Analyze Another Image
            </button>
          </div>

        </div>
      )}
    </section>
  );
}
