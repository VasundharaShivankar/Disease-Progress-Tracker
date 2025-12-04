import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar navbar-expand-lg navbar-light">
      <div className="container-fluid">
        <Link className="navbar-brand d-flex align-items-center" to="/">
          <span className="fw-bold fs-4" style={{ color: '#FF6B6B' }}>🏥</span>
          <span className="ms-2 fw-bold text-dark">Health Plus</span>
        </Link>
        <button className="navbar-toggler border-0" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item">
              <Link className={`nav-link fw-semibold ${location.pathname === '/' ? 'text-primary' : 'text-dark'}`} to="/">🏠 Home</Link>
            </li>
            <li className="nav-item">
              <Link className={`nav-link fw-semibold ${location.pathname === '/login' ? 'text-primary' : 'text-dark'}`} to="/login">🔐 Login</Link>
            </li>
            <li className="nav-item">
              <Link className={`nav-link fw-semibold ${location.pathname === '/progress-tracker' ? 'text-primary' : 'text-dark'}`} to="/progress-tracker">📊 Progress Tracker</Link>
            </li>
            <li className="nav-item">
              <Link className={`nav-link fw-semibold ${location.pathname === '/skin-analysis' ? 'text-primary' : 'text-dark'}`} to="/skin-analysis">🩺 Skin Analysis</Link>
            </li>
            <li className="nav-item">
              <Link className={`nav-link fw-semibold ${location.pathname === '/spin-analysis' ? 'text-primary' : 'text-dark'}`} to="/spin-analysis">🦴 Spine Analysis</Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
