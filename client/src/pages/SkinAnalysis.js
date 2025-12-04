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
                <div className="mt-5">
                  {result.error ? (
                    <div className="alert alert-danger text-center">
                      <span style={{ fontSize: '2rem' }}>❌</span>
                      <h5 className="mt-2">Analysis Failed</h5>
                      <p className="mb-0">{result.error}</p>
                    </div>
                  ) : (
                    <div className="row g-4">
                      <div className="col-12">
                        <div className="card border-0 shadow-sm">
                          <div className="card-body text-center p-4">
                            <span style={{ fontSize: '3rem', color: '#FF6B6B' }}>🚨</span>
                            <h4 className="mt-3 fw-bold">Analysis Result</h4>
                            <div className="bg-light p-3 rounded mt-3">
                              <h5 className="text-primary mb-0">{result.prediction}</h5>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="col-md-6">
                        <div className="card h-100 border-0 shadow-sm">
                          <div className="card-body">
                            <span style={{ fontSize: '2rem', color: '#4ECDC4' }}>🧾</span>
                            <h5 className="card-title mt-2 fw-bold">Explanation</h5>
                            <p className="card-text text-muted">{result.explanation}</p>
                          </div>
                        </div>
                      </div>

                      {result.tips?.length > 0 && (
                        <div className="col-md-6">
                          <div className="card h-100 border-0 shadow-sm">
                            <div className="card-body">
                              <span style={{ fontSize: '2rem', color: '#667eea' }}>🩺</span>
                              <h5 className="card-title mt-2 fw-bold">Health Tips</h5>
                              <ul className="list-unstyled">
                                {result.tips.map((tip, idx) => (
                                  <li key={idx} className="mb-2">
                                    <span style={{ color: '#28a745' }}>✅</span> {tip}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </div>
                      )}

                      {result.advice && (
                        <div className="col-12">
                          <div className="card border-0 shadow-sm">
                            <div className="card-body">
                              <span style={{ fontSize: '2rem', color: '#FF6B6B' }}>📌</span>
                              <h5 className="card-title mt-2 fw-bold">Medical Advice</h5>
                              <p className="card-text text-muted">{result.advice}</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
  );
}

export default SkinAnalysis;
