import React from 'react'
import { TrendingUp, TrendingDown, Target, Zap } from 'lucide-react'

const SEGMENT_COLORS = {
  Cold:    'var(--cold)',
  Warm:    'var(--warm)',
  Hot:     'var(--hot)',
  Convert: 'var(--convert)',
}

const URGENCY_COLORS = {
  low:      'var(--cold)',
  medium:   'var(--warm)',
  high:     'var(--hot)',
  critical: 'var(--convert)',
}

export default function ResultCard({ result, loading }) {
  if (loading) return (
    <div style={cardStyle}>
      <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--muted)' }}>
        <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⟳</div>
        <div className='mono' style={{ fontSize: '0.8rem', letterSpacing: '0.08em' }}>
          RUNNING INFERENCE...
        </div>
      </div>
    </div>
  )

  if (!result) return null

  if (result.error) return (
    <div style={cardStyle}>
      <div style={{ color: 'var(--danger)', padding: '1rem' }}>{result.error}</div>
    </div>
  )

  const { will_purchase, purchase_probability, confidence, recommendation } = result
  const seg = recommendation?.segment
  const segColor = SEGMENT_COLORS[seg] || 'var(--text)'
  const pct = Math.round(purchase_probability * 100)

  return (
    <div style={cardStyle}>
      <div className='mono' style={{
        color: 'var(--accent2)',
        fontSize: '0.7rem',
        letterSpacing: '0.1em',
        marginBottom: '1.25rem',
      }}>PREDICTION_OUTPUT</div>

      {/* Probability Ring */}
      <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
        <div style={{ position: 'relative', display: 'inline-block' }}>
          <svg width='140' height='140' viewBox='0 0 140 140'>
            <circle cx='70' cy='70' r='58' fill='none'
              stroke='var(--border)' strokeWidth='8' />
            <circle cx='70' cy='70' r='58' fill='none'
              stroke={will_purchase ? 'var(--convert)' : 'var(--danger)'}
              strokeWidth='8'
              strokeDasharray={`${2 * Math.PI * 58}`}
              strokeDashoffset={`${2 * Math.PI * 58 * (1 - purchase_probability)}`}
              strokeLinecap='round'
              transform='rotate(-90 70 70)'
              style={{ transition: 'stroke-dashoffset 0.8s ease' }}
            />
          </svg>
          <div style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%,-50%)', textAlign: 'center',
          }}>
            <div style={{
              fontSize: '2rem', fontWeight: 800,
              color: will_purchase ? 'var(--convert)' : 'var(--danger)',
            }}>{pct}%</div>
            <div className='mono' style={{ fontSize: '0.65rem', color: 'var(--muted)' }}>
              PROBABILITY
            </div>
          </div>
        </div>

        <div style={{ marginTop: '0.75rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
          {will_purchase
            ? <TrendingUp size={18} color='var(--convert)' />
            : <TrendingDown size={18} color='var(--danger)' />
          }
          <span style={{
            fontWeight: 700, fontSize: '1rem',
            color: will_purchase ? 'var(--convert)' : 'var(--danger)',
          }}>
            {will_purchase ? 'WILL PURCHASE' : 'WILL NOT PURCHASE'}
          </span>
        </div>
        <div className='mono' style={{ fontSize: '0.7rem', color: 'var(--muted)', marginTop: '4px' }}>
          Confidence: {confidence}
        </div>
      </div>

      {/* Segment Badge */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 14px',
        background: 'var(--bg)',
        border: `1px solid ${segColor}`,
        borderRadius: '8px',
        marginBottom: '1rem',
      }}>
        <div>
          <div className='mono' style={{ fontSize: '0.65rem', color: 'var(--muted)', letterSpacing: '0.08em' }}>
            VISITOR SEGMENT
          </div>
          <div style={{ fontWeight: 700, color: segColor, fontSize: '1.1rem', marginTop: '2px' }}>
            {seg}
          </div>
        </div>
        <Target size={22} color={segColor} />
      </div>

      {/* Recommendation */}
      {recommendation && (
        <div style={{
          padding: '14px',
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          borderLeft: `3px solid ${URGENCY_COLORS[recommendation.urgency]}`,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div className='mono' style={{ fontSize: '0.65rem', color: 'var(--muted)', letterSpacing: '0.08em' }}>
              RECOMMENDED ACTION
            </div>
            <span style={{
              fontSize: '0.65rem', fontWeight: 700,
              color: URGENCY_COLORS[recommendation.urgency],
              background: `${URGENCY_COLORS[recommendation.urgency]}18`,
              padding: '2px 8px', borderRadius: '20px',
              fontFamily: 'DM Mono, monospace',
              letterSpacing: '0.06em',
            }}>
              {recommendation.urgency?.toUpperCase()}
            </span>
          </div>

          <div style={{ fontWeight: 700, fontSize: '0.95rem', margin: '6px 0 4px', color: 'var(--text)' }}>
            {recommendation.action_label}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--muted)', lineHeight: 1.5 }}>
            {recommendation.reason}
          </div>

          {recommendation.confidence_score > 0 && (
            <div className='mono' style={{
              marginTop: '10px', fontSize: '0.65rem',
              color: 'var(--muted)', borderTop: '1px solid var(--border)', paddingTop: '8px',
            }}>
              <Zap size={10} style={{ verticalAlign: 'middle', marginRight: 4 }} />
              BANDIT UCB SCORE: {recommendation.confidence_score.toFixed(3)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const cardStyle = {
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: '12px',
  padding: '1.5rem',
}