import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Training from './pages/Training';
import Forecasts from './pages/Forecasts';
import CompleteVisualization from './pages/CompleteVisualization';
import DatabaseView from './pages/DatabaseView';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/training" element={<Training />} />
            <Route path="/forecasts" element={<Forecasts />} />
            <Route path="/complete" element={<CompleteVisualization />} />
            <Route path="/database" element={<DatabaseView />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;


