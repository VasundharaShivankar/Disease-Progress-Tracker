import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import ProgressTracker from './pages/ProgressTracker';
import SkinAnalysis from './pages/SkinAnalysis';
import SpinAnalysis from './pages/SpinAnalysis';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/progress-tracker" element={<ProgressTracker />} />
          <Route path="/skin-analysis" element={<SkinAnalysis />} />
          <Route path="/spin-analysis" element={<SpinAnalysis />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
