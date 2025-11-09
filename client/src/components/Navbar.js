import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-primary">
      <div className="container-fluid">
        <Link className="navbar-brand" to="/">Disease Progress Tracker</Link>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav">
            <li className="nav-item">
              <Link className={`nav-link ${location.pathname === '/' ? 'active' : ''}`} to="/">Home</Link>
            </li>
            <li className="nav-item">
              <Link className={`nav-link ${location.pathname === '/login' ? 'active' : ''}`} to="/login">Login</Link>
            </li>
            <li className="nav-item">
              <Link className={`nav-link ${location.pathname === '/progress-tracker' ? 'active' : ''}`} to="/progress-tracker">Progress Tracker</Link>
            </li>
            <li className="nav-item">
              <Link className={`nav-link ${location.pathname === '/skin-analysis' ? 'active' : ''}`} to="/skin-analysis">Skin Analysis</Link>
            </li>
            <li className="nav-item">
              <Link className={`nav-link ${location.pathname === '/spin-analysis' ? 'active' : ''}`} to="/spin-analysis">Spin Analysis</Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
