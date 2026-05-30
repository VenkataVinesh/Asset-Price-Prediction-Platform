import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { Cpu, Activity, TrendingUp, Sliders, RefreshCw, Layers } from 'lucide-react';

const Dashboard = () => {
  const [ticker, setTicker] = useState('AAPL');
  const [model, setModel] = useState('lstm');
  const [horizon, setHorizon] = useState(10);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchForecast = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(
        `http://localhost:8000/api/forecast?ticker=${ticker}&model=${model}&horizon=${horizon}`
      );
      if (!response.ok) {
        throw new Error('API server returned an error');
      }
      const json = await response.ok ? await response.json() : null;
      setData(json);
    } catch (err) {
      setError('Could not connect to FastAPI server. Please ensure the backend is running on port 8000.');
      console.error(err);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchForecast();
  }, [ticker, model, horizon]);

  const handleRefresh = () => {
    setIsRefreshing(true);
    fetchForecast();
  };

  if (loading && !isRefreshing) {
    return (
      <div className="status-container">
        <div className="spinner" />
        <span>Instantiating forecast engine and processing indicators...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="status-container" style={{ color: '#ef4444', flexDirection: 'column', gap: '1rem' }}>
        <span>⚠️ Connection Error</span>
        <p className="text-sm" style={{ maxWidth: '500px', textAlign: 'center', color: 'var(--text-sub)' }}>
          {error}
        </p>
        <button className="btn" onClick={fetchForecast} style={{ marginTop: '1rem' }}>
          Retry Connection
        </button>
      </div>
    );
  }

  // Format data for Recharts
  const history = data?.history || [];
  const forecast = data?.forecast || [];
  
  // Combine history and forecast into a single sequence
  const chartData = [
    ...history.map(item => ({
      date: item.date,
      close: item.close,
      sma: item.sma,
      rsi: item.rsi,
      type: 'Historical'
    })),
    // Link the last historical price with the first forecast point
    ...forecast.map((item, index) => ({
      date: item.date,
      forecast: item.price,
      lower: item.lower_bound,
      upper: item.upper_bound,
      type: 'Forecast'
    }))
  ];

  // Align forecast starting point visually with the last historical point
  if (history.length > 0 && chartData.length > history.length) {
    const lastHist = history[history.length - 1];
    chartData[history.length] = {
      ...chartData[history.length],
      close: lastHist.close,
      forecast: lastHist.close,
      lower: lastHist.close,
      upper: lastHist.close
    };
  }

  // Compute quick metrics
  const lastPrice = history[history.length - 1]?.close || 0;
  const targetPrice = forecast[forecast.length - 1]?.price || 0;
  const priceChange = targetPrice - lastPrice;
  const pctChange = (priceChange / lastPrice) * 100;
  const forecastAvg = forecast.reduce((acc, f) => acc + f.price, 0) / (forecast.length || 1);

  return (
    <div className="dashboard-grid">
      
      {/* Parameters Panel */}
      <aside className="panel control-section">
        <div className="flex items-center gap-2" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem' }}>
          <Sliders size={18} className="text-accent-cyan" />
          <h2 style={{ fontSize: '1rem', fontWeight: 700 }}>Forecast Parameters</h2>
        </div>

        <div className="control-group">
          <label>Asset Ticker</label>
          <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
            <option value="AAPL">AAPL (Apple Inc.)</option>
            <option value="MSFT">MSFT (Microsoft Corp.)</option>
            <option value="BTC-USD">BTC-USD (Bitcoin)</option>
          </select>
        </div>

        <div className="control-group">
          <label>Inference Engine</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="lstm">Stacked LSTM (Deep Learning)</option>
            <option value="arima">ARIMA(1, 1, 1) (Statistical)</option>
          </select>
        </div>

        <div className="control-group">
          <label>Forecast Horizon</label>
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
            <option value={5}>5 Steps (Short-term)</option>
            <option value={10}>10 Steps (Medium-term)</option>
            <option value={15}>15 Steps (Extended)</option>
            <option value={30}>30 Steps (Long-term)</option>
          </select>
        </div>

        <button className="btn" onClick={handleRefresh} disabled={isRefreshing} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
          <RefreshCw size={16} className={isRefreshing ? 'animate-spin' : ''} />
          <span>{isRefreshing ? 'Computing...' : 'Recalculate Models'}</span>
        </button>

        <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
          <div className="tagline" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
            <Activity size={12} className="text-accent-teal" />
            <span>Telemetry Online</span>
          </div>
        </div>
      </aside>

      {/* Main Charts & Telemetry Area */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        
        {/* Quick Metrics Bar */}
        <div className="panel metrics-row">
          <div className="metric-card">
            <span className="metric-label">Last Closed Price</span>
            <div className="metric-value">
              ${lastPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          <div className="metric-card">
            <span className="metric-label">Projected Return</span>
            <div className="metric-value" style={{ color: pctChange >= 0 ? '#10b981' : '#ef4444' }}>
              {pctChange >= 0 ? '+' : ''}{pctChange.toFixed(2)}%
            </div>
          </div>
          <div className="metric-card">
            <span className="metric-label">Forecast Avg Price</span>
            <div className="metric-value">
              ${forecastAvg.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          <div className="metric-card">
            <span className="metric-label">Residual Volatility</span>
            <div className="metric-value" style={{ color: 'var(--accent-cyan)' }}>
              {(history[history.length - 1]?.volatility * 100).toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Price Prediction Chart Panel */}
        <div className="panel chart-container">
          <div className="chart-header">
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <TrendingUp size={20} className="text-accent-cyan" />
              <span>Asset Price Prediction Bounds</span>
            </h3>
            <span className="metric-pill">
              Model: {model.toUpperCase()} | Ticker: {ticker}
            </span>
          </div>

          <div style={{ width: '100%', height: '380px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorGlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0.01}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                <XAxis 
                  dataKey="date" 
                  stroke="var(--text-sub)" 
                  tick={{ fill: 'var(--text-sub)', fontSize: 10 }}
                  axisLine={false}
                />
                <YAxis 
                  domain={['auto', 'auto']}
                  stroke="var(--text-sub)"
                  tick={{ fill: 'var(--text-sub)', fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip 
                  contentStyle={{ 
                    background: 'var(--bg-main)', 
                    border: '1px solid var(--border-color)',
                    borderRadius: '0.5rem',
                    color: 'var(--text-main)',
                    fontSize: '0.8rem'
                  }} 
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                
                {/* 95% Confidence Bounds Area */}
                <Area 
                  type="monotone" 
                  dataKey="upper" 
                  stroke="none" 
                  fill="rgba(6, 182, 212, 0.06)" 
                  name="95% Upper Bound"
                  legendType="none"
                />
                <Area 
                  type="monotone" 
                  dataKey="lower" 
                  stroke="none" 
                  fill="var(--bg-main)" 
                  name="95% Lower Bound"
                  legendType="none"
                />

                {/* Shaded area between bounds */}
                <Area 
                  type="monotone" 
                  dataKey="upper" 
                  stroke="none" 
                  fill="url(#colorGlow)" 
                  name="Confidence Envelope"
                />

                {/* Historical Price */}
                <Line 
                  type="monotone" 
                  dataKey="close" 
                  stroke="var(--accent-blue)" 
                  strokeWidth={2.5} 
                  dot={false}
                  name="Historical Price"
                  activeDot={{ r: 4 }}
                />

                {/* Technical Indicator - SMA */}
                <Line 
                  type="monotone" 
                  dataKey="sma" 
                  stroke="var(--accent-teal)" 
                  strokeWidth={1.5} 
                  dot={false}
                  strokeDasharray="4 4"
                  name="Rolling SMA (14)"
                  opacity={0.6}
                />

                {/* Forecast Price */}
                <Line 
                  type="monotone" 
                  dataKey="forecast" 
                  stroke="var(--accent-cyan)" 
                  strokeWidth={2.5} 
                  dot={false}
                  name="Forecasted Trend"
                  activeDot={{ r: 4 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Technical Architecture & Mathematics Explanations */}
        <div className="tech-details">
          
          <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={16} className="text-accent-teal" />
              <span>Feature Pipeline Technical Specs</span>
            </h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-sub)', leadingRelaxed: true }}>
              Historical daily price telemetry is continuously ingested by the FastAPI pipeline. On each inference request, the backend calculates:
            </p>
            <ul style={{ fontSize: '0.8rem', color: 'var(--text-sub)', display: 'flex', flexDirection: 'column', gap: '0.5rem', paddingLeft: '1.25rem' }}>
              <li><strong>RSI (14 Days):</strong> Analyzes momentum boundaries to evaluate oversold/overbought thresholds.</li>
              <li><strong>EMA & SMA:</strong> Smooths noise variance to identify multi-step directional trend channels.</li>
              <li><strong>Rolling Volatility:</strong> Computes the standard deviation of log returns over a 14-day rolling window to scale confidence bounds.</li>
            </ul>
          </div>

          <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu size={16} className="text-accent-cyan" />
              <span>Inference Layer Execution</span>
            </h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-sub)', leadingRelaxed: true }}>
              Depending on the chosen engine, the platform executes different algorithms:
            </p>
            <ul style={{ fontSize: '0.8rem', color: 'var(--text-sub)', display: 'flex', flexDirection: 'column', gap: '0.5rem', paddingLeft: '1.25rem' }}>
              <li><strong>Deep Stacked LSTM:</strong> Uses PyTorch sequence projections to loop predictions recursively back into the input state vector, tracking temporal dependencies.</li>
              <li><strong>Classical ARIMA:</strong> Fits autoregressive (p) and moving average (q) terms on pricing difference (d) equations, providing analytical 95% confidence intervals.</li>
            </ul>
          </div>

        </div>

      </div>

    </div>
  );
};

export default Dashboard;
