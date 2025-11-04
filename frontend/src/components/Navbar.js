import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          ⚡ Energy Forecast
        </Link>
        <div className="navbar-links">
          <Link 
            to="/" 
            className={location.pathname === '/' ? 'active' : ''}
          >
            Dashboard
          </Link>
          <Link 
            to="/training" 
            className={location.pathname === '/training' ? 'active' : ''}
          >
            Training
          </Link>
          <Link 
            to="/forecasts" 
            className={location.pathname === '/forecasts' ? 'active' : ''}
          >
            Forecasts
          </Link>
          <Link 
            to="/complete" 
            className={location.pathname === '/complete' ? 'active' : ''}
          >
            Complete View
          </Link>
          <Link 
            to="/database" 
            className={location.pathname === '/database' ? 'active' : ''}
          >
            Database
          </Link>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;


