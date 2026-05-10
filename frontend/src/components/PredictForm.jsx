import React, { useState } from 'react'
import { Zap } from 'lucide-react'

const DEFAULTS = {
  Administrative: 2,
  Administrative_Duration: 80,
  Informational: 0,
  Informational_Duration: 0,
  ProductRelated: 12,
  ProductRelated_Duration: 720,
  BounceRates: 0.02,
  ExitRates: 0.04,
  PageValues: 25,
  SpecialDay: 0,
  Month: 11,
  OperatingSystems: 2,
  Browser: 2,
  Region: 1,
  TrafficType: 2,
  VisitorType: 2,
  Weekend: 0,
}

const FIELDS = [
  { key: 'ProductRelated',          label: 'Product Pages Visited',    min: 0,  max: 100, step: 1 },
  { key: 'ProductRelated_Duration', label: 'Time on Product Pages (s)',min: 0,  max: 5000,step: 10 },
  { key: 'PageValues',              label: 'Page Value Score',         min: 0,  max: 400, step: 1 },
  { key: 'BounceRates',             label: 'Bounce Rate',              min: 0,  max: 1,   step: 0.01 },
  { key: 'ExitRates',               label: 'Exit Rate',                min: 0,  max: 1,   step: 0.01 },
  { key: 'Administrative',          label: 'Admin Pages Visited',      min: 0,  max: 30,  step: 1 },
  { key: 'Administrative_Duration', label: 'Time on Admin Pages (s)',  min: 0,  max: 3000,step: 10 },
  { key: 'Month',                   label: 'Month (1–12)',             min: 1,  max: 12,  step: 1 },
  { key: 'SpecialDay',              label: 'Special Day Proximity',    min: 0,  max: 1,   step: 0.1 },
  { key: 'TrafficType',             label: 'Traffic Type',             min: 1,  max: 20,  step: 1 },
  { key: 'Region',                  label: 'Region',                   min: 1,  max: 9,   step: 1 },
]

const VISITOR_TYPES = { 0: 'New Visitor', 1: 'Other', 2: 'Returning' }

function deriveEngineered(f) {
  const totalPages    = f.Administrative + f.Informational + f.ProductRelated
  const totalDuration = f.Administrative_Duration + f.Informational_Duration + f.ProductRelated_Duration
  return {
    TotalPages:         totalPages,
    TotalDuration:      totalDuration,
    ProductPageRatio:   totalPages > 0 ? f.ProductRelated / totalPages : 0,
    AvgTimePerPage:     totalPages > 0 ? totalDuration / totalPages : 0,
    HighPageValue:      f.PageValues >= 25 ? 1 : 0,
    NearSpecialDay:     f.SpecialDay > 0 ? 1 : 0,
    ExitBounceRisk:     f.BounceRates + f.ExitRates,
    Informational:      f.Informational,
    Informational_Duration: f.Informational_Duration,
    OperatingSystems:   f.OperatingSystems,
    Browser:            f.Browser,
  }
}

export default function PredictForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS)

  const set = (key, val) => setForm(f => ({ ...f, [key]: Number(val) }))

  const handleSubmit = () => {
    const engineered = deriveEngineered(form)
    onSubmit({ ...form, ...engineered })
  }

  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '12px',
      padding: '1.5rem',
    }}>
      <div className='mono' style={{
        color: 'var(--accent)',
        fontSize: '0.7rem',
        letterSpacing: '0.1em',
        marginBottom: '1.25rem',
      }}>SESSION_FEATURES</div>

      {/* Visitor Type + Weekend */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        <div>
          <label style={labelStyle}>Visitor Type</label>
          <select
            value={form.VisitorType}
            onChange={e => set('VisitorType', e.target.value)}
            style={selectStyle}
          >
            {Object.entries(VISITOR_TYPES).map(([v, l]) => (
              <option key={v} value={v}>{l}</option>
            ))}
          </select>
        </div>
        <div>
          <label style={labelStyle}>Weekend Session</label>
          <select
            value={form.Weekend}
            onChange={e => set('Weekend', e.target.value)}
            style={selectStyle}
          >
            <option value={0}>No</option>
            <option value={1}>Yes</option>
          </select>
        </div>
      </div>

      {/* Sliders */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
        {FIELDS.map(({ key, label, min, max, step }) => (
          <div key={key}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
              <label style={labelStyle}>{label}</label>
              <span className='mono' style={{ fontSize: '0.75rem', color: 'var(--accent)' }}>
                {form[key]}
              </span>
            </div>
            <input
              type='range'
              min={min} max={max} step={step}
              value={form[key]}
              onChange={e => set(key, e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
            />
          </div>
        ))}
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{
          marginTop: '1.5rem',
          width: '100%',
          padding: '12px',
          background: loading ? 'var(--border)' : 'var(--accent)',
          color: '#0a0a0f',
          border: 'none',
          borderRadius: '8px',
          fontFamily: 'Syne, sans-serif',
          fontWeight: 800,
          fontSize: '0.9rem',
          letterSpacing: '0.06em',
          cursor: loading ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          transition: 'all 0.15s',
        }}
      >
        <Zap size={16} />
        {loading ? 'PREDICTING...' : 'PREDICT INTENT'}
      </button>
    </div>
  )
}

const labelStyle = {
  fontSize: '0.75rem',
  color: 'var(--muted)',
  fontWeight: 600,
  letterSpacing: '0.04em',
  display: 'block',
}

const selectStyle = {
  width: '100%',
  marginTop: '4px',
  padding: '8px 10px',
  background: 'var(--bg)',
  border: '1px solid var(--border)',
  borderRadius: '6px',
  color: 'var(--text)',
  fontFamily: 'Syne, sans-serif',
  fontSize: '0.85rem',
  cursor: 'pointer',
}