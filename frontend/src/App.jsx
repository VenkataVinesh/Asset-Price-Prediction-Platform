import React from 'react'
import Dashboard from './components/Dashboard'

function App() {
  return (
    <div className="app-container">
      <header>
        <div>
          <h1 className="logo">Asset Price Prediction Platform</h1>
          <p className="tagline">Machine Learning & Statistical Forecasting Core</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="tagline" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981', display: 'inline-block', animate: 'pulse' }} />
            <span>Forecasting Engine Active</span>
          </div>
        </div>
      </header>

      <main>
        <Dashboard />
      </main>
    </div>
  )
}

export default App
