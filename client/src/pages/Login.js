import React, { useState } from 'react';

const Login = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle login logic here
    console.log('Login data:', formData);
  };

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-6 col-lg-5">
          <div className="card shadow-lg border-0">
            <div className="card-body p-5">
              <div className="text-center mb-4">
                <span style={{ fontSize: '4rem', color: '#FF6B6B' }}>🔐</span>
                <h2 className="mt-3 fw-bold" style={{ color: '#333' }}>Welcome Back</h2>
                <p className="text-muted">Sign in to access Health Plus</p>
              </div>

              <form onSubmit={handleSubmit}>
                <div className="mb-4">
                  <label htmlFor="username" className="form-label fw-semibold">👤 Username</label>
                  <input
                    type="text"
                    className="form-control form-control-lg"
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    placeholder="Enter your username"
                    required
                  />
                </div>
                <div className="mb-4">
                  <label htmlFor="password" className="form-label fw-semibold">🔒 Password</label>
                  <input
                    type="password"
                    className="form-control form-control-lg"
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="Enter your password"
                    required
                  />
                </div>
                <div className="d-grid">
                  <button type="submit" className="btn btn-primary btn-lg fw-semibold">
                    🚀 Sign In
                  </button>
                </div>
              </form>

              <div className="text-center mt-4">
                <p className="text-muted mb-0">Don't have an account? <a href="#" className="text-primary fw-semibold">Contact Admin</a></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
