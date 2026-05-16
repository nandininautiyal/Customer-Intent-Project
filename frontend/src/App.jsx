import React, { useState, useEffect } from 'react'
import PredictForm from './components/PredictForm'
import ResultCard from './components/ResultCard'
import MetricsDashboard from './components/MetricsDashboard'
import { Brain } from 'lucide-react'

const NAV = ['Predict', 'Dashboard']


export const API_BASE = import.meta.env.VITE_API_URL ?? ''

export default function App() {
  const [tab, setTab]         = useState('Predict')
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setApiStatus(d.status === 'ok' ? 'live' : 'error'))
      .catch(() => setApiStatus('error'))
  }, [])

  const handlePredict = async (formData) => {
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setResult({ error: 'API unreachable. Make sure the FastAPI server is running.' })
    }
    setLoading(false)
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>

      {/* Header */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        padding: '0 2rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '60px',
        position: 'sticky',
        top: 0,
        background: 'var(--bg)',
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Brain size={20} color='var(--accent)' />
          <span style={{
            fontWeight: 800,
            fontSize: '1rem',
            letterSpacing: '-0.02em',
            color: 'var(--text)'
          }}>
            INTENT<span style={{ color: 'var(--accent)' }}>ENGINE</span>
          </span>
        </div>

        <nav style={{ display: 'flex', gap: '4px' }}>
          {NAV.map(n => (
            <button key={n} onClick={() => setTab(n)} style={{
              background: tab === n ? 'var(--border)' : 'transparent',
              border: 'none',
              color: tab === n ? 'var(--accent)' : 'var(--muted)',
              padding: '6px 16px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontFamily: 'Syne, sans-serif',
              fontWeight: 600,
              fontSize: '0.85rem',
              letterSpacing: '0.04em',
              transition: 'all 0.15s',
            }}>{n.toUpperCase()}</button>
          ))}
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{
            width: 7, height: 7, borderRadius: '50%',
            background: apiStatus === 'live' ? 'var(--convert)' : 'var(--danger)',
            boxShadow: apiStatus === 'live' ? '0 0 8px var(--convert)' : 'none',
          }} />
          <span className='mono' style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
            {apiStatus === 'live' ? 'API LIVE' : 'API OFFLINE'}
          </span>
        </div>
      </header>

      {/* Main content */}
      <div style={{ padding: '3rem 2rem 2rem', maxWidth: '1100px', margin: '0 auto' }}>
        {tab === 'Predict' && (
          <>
            <div style={{ marginBottom: '2.5rem' }}>
              <div className='mono' style={{
                color: 'var(--accent)',
                fontSize: '0.75rem',
                letterSpacing: '0.12em',
                marginBottom: '0.5rem'
              }}>
                // PURCHASE INTENT PREDICTION
              </div>
              <h1 style={{
                fontSize: 'clamp(1.8rem, 4vw, 3rem)',
                fontWeight: 800,
                letterSpacing: '-0.03em',
                lineHeight: 1.1,
                color: 'var(--text)',
              }}>
                Will this visitor<br />
                <span style={{ color: 'var(--accent)' }}>convert?</span>
              </h1>
              <p style={{
                marginTop: '0.75rem',
                color: 'var(--muted)',
                fontSize: '0.95rem',
                maxWidth: '480px',
                lineHeight: 1.6,
              }}>
                Enter session features to get a purchase probability,
                visitor segment, and the optimal marketing intervention
                from the LinUCB bandit recommender.
              </p>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: result ? '1fr 1fr' : '1fr',
              gap: '1.5rem',
              transition: 'all 0.3s',
            }}>
              <PredictForm onSubmit={handlePredict} loading={loading} />
              {(result || loading) && (
                <ResultCard result={result} loading={loading} />
              )}
            </div>
          </>
        )}

        {tab === 'Dashboard' && <MetricsDashboard />}
      </div>
    </div>
  )
}