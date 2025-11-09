import React, { useState } from 'react';
import axios from 'axios';
import './classifier.css';

const SkinAnalysis = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      setResult(response.data);
    } catch (error) {
      console.error(error);
      setResult({ error: 'Failed to get prediction' });
    }
  };

  return (
    <div className="img-upload-container">
      <h2 className="upload-head">Skin Condition Classifier</h2>
      
      <input
        type="file"
        accept="image/*"
        id="fileInput"
        onChange={handleFileChange}
        className="hidden-input"
      />
      <label htmlFor="fileInput" className="custom-file-button">
        📁 Choose File
      </label>

      {preview && (
        <div className="preview-section">
          <h4 className="preview-title">Preview:</h4>
          <img src={preview} alt="Uploaded Preview" className="imgbox" />
        </div>
      )}

      <button className="btn-upload" onClick={handleSubmit}>
        🔍 Predict
      </button>

      {result && (
        <div className="result-container">
          {result.error ? (
            <p className="error-text">{result.error}</p>
          ) : (
            <>
              <div className="prediction-box">
                <p className="prediction-title">🚨 Prediction:</p>
                <p className="prediction-label">{result.prediction}</p>
              </div>

              <div className="explanation-box">
                <p className="section-title">🧾 Explanation:</p>
                <p>{result.explanation}</p>
              </div>

              {result.tips?.length > 0 && (
                <div className="tips-box">
                  <p className="section-title">🩺 Health Tips:</p>
                  <ul>
                    {result.tips.map((tip, idx) => (
                      <li key={idx}>✅ {tip}</li>
                    ))}
                  </ul>
                </div>
              )}

              {result.advice && (
                <div className="advice-box">
                  <p className="section-title">📌 Advice:</p>
                  <p>{result.advice}</p>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default SkinAnalysis;
