import React from 'react';

const SpinAnalysis = () => {
  return (
    <div className="container mt-5">
      <div className="text-center mb-5">
        <span style={{ fontSize: '4rem', color: '#FF6B6B' }}>🦴</span>
        <h1 className="display-4 fw-bold mt-3" style={{ color: '#333' }}>Spine Analysis</h1>
        <p className="lead text-muted">Advanced AI-powered scoliosis detection and spinal curvature measurement</p>
      </div>

      <div className="row g-4 mb-5">
        <div className="col-md-4">
          <div className="card h-100 text-center border-0 shadow-sm">
            <div className="card-body">
              <span style={{ fontSize: '3rem', color: '#4ECDC4' }}>📐</span>
              <h5 className="card-title mt-3 fw-bold">Curvature Measurement</h5>
              <p className="card-text text-muted">Precise Cobb angle calculation for accurate scoliosis assessment</p>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card h-100 text-center border-0 shadow-sm">
            <div className="card-body">
              <span style={{ fontSize: '3rem', color: '#667eea' }}>🔍</span>
              <h5 className="card-title mt-3 fw-bold">AI Detection</h5>
              <p className="card-text text-muted">Machine learning algorithms for automatic spine detection and analysis</p>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card h-100 text-center border-0 shadow-sm">
            <div className="card-body">
              <span style={{ fontSize: '3rem', color: '#FF6B6B' }}>📊</span>
              <h5 className="card-title mt-3 fw-bold">Progress Tracking</h5>
              <p className="card-text text-muted">Monitor treatment progress with detailed analysis reports</p>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-4 mb-5">
        <div className="col-md-6">
          <div className="card h-100 border-0 shadow-sm">
            <div className="card-body">
              <span style={{ fontSize: '2rem', color: '#4ECDC4' }}>🏥</span>
              <h5 className="card-title mt-2 fw-bold">Clinical Applications</h5>
              <ul className="list-unstyled">
                <li className="mb-2"><span style={{ color: '#4ECDC4' }}>•</span> Early scoliosis detection</li>
                <li className="mb-2"><span style={{ color: '#4ECDC4' }}>•</span> Posture assessment</li>
                <li className="mb-2"><span style={{ color: '#4ECDC4' }}>•</span> Treatment monitoring</li>
                <li className="mb-0"><span style={{ color: '#4ECDC4' }}>•</span> Surgical planning</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card h-100 border-0 shadow-sm">
            <div className="card-body">
              <span style={{ fontSize: '2rem', color: '#667eea' }}>⚡</span>
              <h5 className="card-title mt-2 fw-bold">Technology Features</h5>
              <ul className="list-unstyled">
                <li className="mb-2"><span style={{ color: '#667eea' }}>•</span> Real-time analysis</li>
                <li className="mb-2"><span style={{ color: '#667eea' }}>•</span> High accuracy detection</li>
                <li className="mb-2"><span style={{ color: '#667eea' }}>•</span> Automated measurements</li>
                <li className="mb-0"><span style={{ color: '#667eea' }}>•</span> Detailed visualization</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="card border-0 shadow-sm">
        <div className="card-body p-5 text-center">
          <span style={{ fontSize: '5rem', color: '#667eea' }}>🔬</span>
          <h3 className="mt-3 fw-bold">Advanced Spine Analysis Coming Soon</h3>
          <p className="text-muted mb-4">Our cutting-edge AI system is being developed to provide comprehensive spinal analysis and scoliosis detection with medical-grade accuracy.</p>
          <div className="progress mb-3" style={{ height: '10px' }}>
            <div className="progress-bar bg-gradient" role="progressbar" style={{ width: '75%', background: 'linear-gradient(45deg, #667eea, #FF6B6B)' }} aria-valuenow="75" aria-valuemin="0" aria-valuemax="100"></div>
          </div>
          <p className="text-muted small">Development Progress: 75% Complete</p>
        </div>
      </div>
    </div>
  );
};

export default SpinAnalysis;
