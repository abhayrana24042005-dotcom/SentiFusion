import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import TextInput from './components/TextInput';
import ImageUpload from './components/ImageUpload';
import AnalyzeButton from './components/AnalyzeButton';
import AnalysisStatus from './components/AnalysisStatus';
import './styles/App.css';

const getBackendHost = () => {
  if (typeof window !== 'undefined' && window.location.hostname) {
    return `http://${window.location.hostname}:8000`;
  }
  return 'http://127.0.0.1:8000';
};

export default function App() {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('sentifusion_theme');
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      return 'light';
    }
    return 'dark';
  });

  const [text, setText] = useState('');
  const [selectedSentiment, setSelectedSentiment] = useState('Positive');
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageHash, setImageHash] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statusData, setStatusData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [validationErrors, setValidationErrors] = useState({ text: null, image: null });

  // Sync theme changes to html element and local storage
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('sentifusion_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Manage object URL lifecycle
  useEffect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [imageFile]);

  // Handle selecting an image file
  const handleSelectFile = (file) => {
    setImageFile(file);
    setImageHash(null);
    setValidationErrors((prev) => ({ ...prev, image: null }));
    setStatusData(null);
    setErrorMessage(null);
  };

  // Handle removing the selected image
  const handleRemoveFile = () => {
    setImageFile(null);
    setImageHash(null);
    setValidationErrors((prev) => ({ ...prev, image: null }));
    setStatusData(null);
  };

  // Handle text input changes
  const handleTextChange = (newText) => {
    setText(newText);
    setValidationErrors((prev) => ({ ...prev, text: null }));
  };

  // Instant sentiment rematch if image is already analyzed
  const handleSelectSentiment = async (newSentiment) => {
    setSelectedSentiment(newSentiment);

    // If an image was already analyzed and we have an imageHash, execute instant rematch (<1ms)
    if (imageHash && statusData && statusData.prediction) {
      try {
        const formData = new FormData();
        formData.append('image_hash', imageHash);
        formData.append('selected_sentiment', newSentiment);
        formData.append('custom_text', text);

        const res = await fetch(`${getBackendHost()}/match-sentiment`, {
          method: 'POST',
          body: formData
        });
        const matchData = await res.json();

        if (matchData.success && matchData.match) {
          // Update status data immediately with zero image reprocessing
          setStatusData((prev) => ({
            ...prev,
            performance: {
              ...prev.performance,
              cache_hit: true,
              cache_type: 'memory_rematch',
              processing_time_ms: 0.4
            },
            prediction: {
              ...prev.prediction,
              sentiment_match: matchData.match,
              perspectives: {
                ...prev.prediction.perspectives,
                user_sentiment_perspective: matchData.match.explanation
              }
            }
          }));
        }
      } catch (err) {
        console.warn('Instant sentiment match notice:', err);
      }
    }
  };

  // Reset entire form
  const handleReset = () => {
    setText('');
    setImageFile(null);
    setImageHash(null);
    setStatusData(null);
    setErrorMessage(null);
    setValidationErrors({ text: null, image: null });
  };

  // Validate and submit multimodal payload
  const handleAnalyze = async () => {
    setErrorMessage(null);
    setStatusData(null);

    const errors = { text: null, image: null };
    let hasError = false;

    if (!imageFile) {
      errors.image = 'An image file is required before analyzing.';
      hasError = true;
    }

    setValidationErrors(errors);
    if (hasError) return;

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('text', text.trim());
      formData.append('selected_sentiment', selectedSentiment);
      formData.append('image', imageFile);

      const response = await fetch(`${getBackendHost()}/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.message || `Server responded with status: ${response.status}`);
      }

      setStatusData(data);
      if (data.image_hash) {
        setImageHash(data.image_hash);
      }
    } catch (err) {
      console.error('Analysis request error:', err);
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setErrorMessage(
          'Could not connect to SentiFusion backend at ' + getBackendHost() + '. Please ensure the FastAPI server is running on port 8000.'
        );
      } else {
        setErrorMessage(err.message || 'An unexpected error occurred during multimodal analysis.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container" data-theme={theme}>
      {/* Header with Theme Switcher */}
      <Header theme={theme} onToggleTheme={toggleTheme} />

      {/* Main Multimodal Input Grid */}
      <main className="modality-grid">
        <TextInput
          value={text}
          onChange={handleTextChange}
          selectedSentiment={selectedSentiment}
          onSelectSentiment={handleSelectSentiment}
          disabled={isLoading}
          error={validationErrors.text}
        />

        <ImageUpload
          file={imageFile}
          previewUrl={previewUrl}
          onSelectFile={handleSelectFile}
          onRemoveFile={handleRemoveFile}
          disabled={isLoading}
          error={validationErrors.image}
        />
      </main>

      {/* Primary Action Button */}
      <AnalyzeButton
        onClick={handleAnalyze}
        isLoading={isLoading}
        disabled={isLoading}
      />

      {/* Analysis Status Output */}
      <AnalysisStatus
        isLoading={isLoading}
        statusData={statusData}
        errorMessage={errorMessage}
        onReset={handleReset}
        selectedSentiment={selectedSentiment}
        onSelectSentiment={handleSelectSentiment}
      />
    </div>
  );
}
