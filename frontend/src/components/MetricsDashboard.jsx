import React, { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Cell
} from 'recharts'

const MODEL_METRICS = [
  { model: 'XGBoost',   auc: 0.928, f1: 0.672, precision: 0.631, recall: 0.717, lift: 0.777 },
  { model: 'Ensemble',  auc: 0.920, f1: 0.660, precision: 0.660, recall: 0.660, lift: 0.772 },
  { model: 'NeuralNet', auc: 0.902, f1: 0.639, precision: 0.632, recall: 0.647, lift: 0.744 },
  { model: 'Logistic',  auc: 0.897, f1: 0.628, precision: 0.585, recall: 0.678, lift: 0.730 },
]

const MODEL_COLORS = {
  XGBoost:   '#e8ff47',
  Ensemble:  '#47ffe8',
  NeuralNet: '#47ff8a',
  Logistic:  '#4799ff',
}

const SEG_COLORS = {
  Cold:    'var(--cold)',
  Warm:    'var(--warm)',
  Hot:     'var(--hot)',
  Convert: 'var(--convert)',
}

const tooltipStyle = {
  contentStyle: { background: '#12121a', border: '1px solid #1e1e2e', borderRadius: 8, fontFamily: 'DM Mono' },
  labelStyle:   { color: '#e8e8f0', fontSize: 11 },
  itemStyle:    { color: '#e8ff47', fontSize: 11 },
}

export default function MetricsDashboard() {
  const [banditStats, setBanditStats] = useState(null)
  const [segStats, setSegStats]       = useState(null)
  const [activeMetric, setActiveMetric] = useState('auc')

  useEffect(() => {
    fetch('http://localhost:8000/bandit-stats').then(r => r.json()).then(setBanditStats).catch(() => {})
    fetch('http://localhost:8000/segment-stats').then(r => r.json()).then(d => setSegStats(d.segment_distribution)).catch(() => {})
  }, [])

  const radarData = ['auc', 'f1', 'precision', 'recall', 'lift'].map(k => ({
    metric: k.toUpperCase(),
    XGBoost:   MODEL_METRICS[0][k],
    Ensemble:  MODEL_METRICS[1][k],
    NeuralNet: MODEL_METRICS[2][k],
    Logistic:  MODEL_METRICS[3][k],
  }))

  const banditData = banditStats
    ? Object.entries(banditStats.action_stats).map(([action, stats]) => ({
        action: action.replace('show_', '').replace(/_/g, ' '),
        chosen: stats.times_chosen,
        reward: stats.avg_reward,
      }))
    : []

  const segData = segStats
    ? Object.entries(segStats).map(([seg, d]) => ({
        seg, pct: d.percentage, count: d.count
      }))
    : []

  const METRICS = ['auc', 'f1', 'precision', 'recall', 'lift']

  return (
    <div>
      <div style={{ marginBottom: '2rem' }}>
        <div className='mono' style={{ color: 'var(--accent)', fontSize: '0.75rem', letterSpacing: '0.12em', marginBottom: '0.4rem' }}>
          // MODEL PERFORMANCE
        </div>
        <h2 style={{ fontWeight: 800, fontSize: '1.8rem', letterSpacing: '-0.02em' }}>
          Analytics <span style={{ color: 'var(--accent)' }}>Dashboard</span>
        </h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', marginBottom: '1.25rem' }}>

        {/* Model Comparison Bar */}
        <div style={panelStyle}>
          <div style={panelHeaderStyle}>MODEL COMPARISON</div>
          <div style={{ display: 'flex', gap: '6px', marginBottom: '1rem', flexWrap: 'wrap' }}>
            {METRICS.map(m => (
              <button key={m} onClick={() => setActiveMetric(m)} style={{
                padding: '3px 10px', borderRadius: '20px', border: 'none', cursor: 'pointer',
                fontFamily: 'DM Mono', fontSize: '0.65rem', letterSpacing: '0.06em',
                background: activeMetric === m ? 'var(--accent)' : 'var(--border)',
                color: activeMetric === m ? '#0a0a0f' : 'var(--muted)',
                fontWeight: 700,
              }}>{m.toUpperCase()}</button>
            ))}
          </div>
          <ResponsiveContainer width='100%' height={200}>
            <BarChart data={MODEL_METRICS} margin={{ left: -20 }}>
              <XAxis dataKey='model' tick={{ fill: '#5a5a7a', fontSize: 11, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#5a5a7a', fontSize: 10, fontFamily: 'DM Mono' }} domain={[0.5, 1]} axisLine={false} tickLine={false} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey={activeMetric} radius={[4, 4, 0, 0]}>
                {MODEL_METRICS.map(m => (
                  <Cell key={m.model} fill={MODEL_COLORS[m.model]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar */}
        <div style={panelStyle}>
          <div style={panelHeaderStyle}>MULTI-METRIC RADAR</div>
          <ResponsiveContainer width='100%' height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke='var(--border)' />
              <PolarAngleAxis dataKey='metric' tick={{ fill: '#5a5a7a', fontSize: 10, fontFamily: 'DM Mono' }} />
              <PolarRadiusAxis angle={30} domain={[0.5, 1]} tick={false} axisLine={false} />
              {Object.entries(MODEL_COLORS).map(([model, color]) => (
                <Radar key={model} name={model} dataKey={model}
                  stroke={color} fill={color} fillOpacity={0.08} strokeWidth={2} />
              ))}
            </RadarChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '6px' }}>
            {Object.entries(MODEL_COLORS).map(([m, c]) => (
              <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />
                <span className='mono' style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>{m}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Segment Distribution */}
        <div style={panelStyle}>
          <div style={panelHeaderStyle}>VISITOR SEGMENTS</div>
          {segData.length > 0 ? (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '1rem' }}>
                {segData.map(({ seg, pct, count }) => (
                  <div key={seg} style={{
                    padding: '10px 12px',
                    background: 'var(--bg)',
                    border: `1px solid ${SEG_COLORS[seg]}`,
                    borderRadius: '8px',
                  }}>
                    <div style={{ fontWeight: 700, color: SEG_COLORS[seg], fontSize: '0.85rem' }}>{seg}</div>
                    <div style={{ fontWeight: 800, fontSize: '1.4rem', color: 'var(--text)', lineHeight: 1.2 }}>{pct}%</div>
                    <div className='mono' style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>{count} sessions</div>
                  </div>
                ))}
              </div>
              <ResponsiveContainer width='100%' height={100}>
                <BarChart data={segData} margin={{ left: -20 }}>
                  <XAxis dataKey='seg' tick={{ fill: '#5a5a7a', fontSize: 10, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
                  <YAxis hide />
                  <Tooltip {...tooltipStyle} formatter={(v) => [`${v}%`, 'Share']} />
                  <Bar dataKey='pct' radius={[4, 4, 0, 0]}>
                    {segData.map(({ seg }) => <Cell key={seg} fill={SEG_COLORS[seg]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </>
          ) : (
            <div className='mono' style={{ color: 'var(--muted)', fontSize: '0.8rem', padding: '2rem 0', textAlign: 'center' }}>
              Start the API to load live segment data
            </div>
          )}
        </div>

        {/* Bandit Stats */}
        <div style={panelStyle}>
          <div style={panelHeaderStyle}>BANDIT ACTION DISTRIBUTION</div>
          {banditData.length > 0 ? (
            <>
              <div className='mono' style={{ fontSize: '0.7rem', color: 'var(--muted)', marginBottom: '0.75rem' }}>
                Total predictions: {banditStats?.total_predictions ?? 0}
              </div>
              <ResponsiveContainer width='100%' height={200}>
                <BarChart data={banditData} layout='vertical' margin={{ left: 0, right: 10 }}>
                  <XAxis type='number' tick={{ fill: '#5a5a7a', fontSize: 9, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
                  <YAxis type='category' dataKey='action' width={110}
                    tick={{ fill: '#5a5a7a', fontSize: 9, fontFamily: 'DM Mono' }} axisLine={false} tickLine={false} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey='chosen' radius={[0, 4, 4, 0]} fill='var(--accent2)' />
                </BarChart>
              </ResponsiveContainer>
            </>
          ) : (
            <div className='mono' style={{ color: 'var(--muted)', fontSize: '0.8rem', padding: '2rem 0', textAlign: 'center' }}>
              Start the API to load bandit stats
            </div>
          )}
        </div>

      </div>

      {/* Metrics Table */}
      <div style={panelStyle}>
        <div style={panelHeaderStyle}>FULL METRICS TABLE</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'DM Mono', fontSize: '0.8rem' }}>
            <thead>
              <tr>
                {['Model', 'ROC-AUC', 'F1', 'Precision', 'Recall', 'Lift@20%'].map(h => (
                  <th key={h} style={{ padding: '8px 12px', color: 'var(--muted)', fontWeight: 500, textAlign: 'left', borderBottom: '1px solid var(--border)', letterSpacing: '0.06em', fontSize: '0.7rem' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MODEL_METRICS.map((m, i) => (
                <tr key={m.model} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '10px 12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 8, height: 8, borderRadius: 2, background: MODEL_COLORS[m.model] }} />
                      <span style={{ color: 'var(--text)', fontWeight: 600 }}>{m.model}</span>
                      {i === 0 && <span style={{ fontSize: '0.6rem', color: 'var(--accent)', background: '#e8ff4718', padding: '1px 6px', borderRadius: 10 }}>BEST AUC</span>}
                      {i === 1 && <span style={{ fontSize: '0.6rem', color: 'var(--accent2)', background: '#47ffe818', padding: '1px 6px', borderRadius: 10 }}>DEPLOYED</span>}
                    </div>
                  </td>
                  {['auc', 'f1', 'precision', 'recall', 'lift'].map(k => (
                    <td key={k} style={{ padding: '10px 12px', color: 'var(--text)' }}>{m[k].toFixed(3)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

const panelStyle = {
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: '12px',
  padding: '1.25rem',
}

const panelHeaderStyle = {
  fontFamily: 'DM Mono, monospace',
  fontSize: '0.65rem',
  letterSpacing: '0.12em',
  color: 'var(--muted)',
  marginBottom: '1rem',
}