import React from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { 
  ShieldAlert, 
  Activity, 
  Clock, 
  CheckCircle2, 
  AlertTriangle, 
  TrendingUp, 
  ArrowRight,
  RefreshCw,
  Camera
} from 'lucide-react'
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from 'recharts'
import { fetchSummary, fetchTrends, fetchViolations } from '../api'

const COLORS = ['#ef4444', '#f97316', '#dc2626', '#a855f7', '#eab308', '#6b7280']

export default function Dashboard() {
  const { 
    data: summary, 
    isLoading: summaryLoading, 
    refetch: refetchSummary 
  } = useQuery({
    queryKey: ['summary'],
    queryFn: fetchSummary,
    refetchInterval: 10_000,
  })

  const { 
    data: trends, 
    isLoading: trendsLoading 
  } = useQuery({
    queryKey: ['trends'],
    queryFn: () => fetchTrends(7),
    refetchInterval: 30_000,
  })

  const { 
    data: recentViolations, 
    isLoading: violationsLoading 
  } = useQuery({
    queryKey: ['recentViolations'],
    queryFn: () => fetchViolations({ page: 1, page_size: 5 }),
    refetchInterval: 5000, // Fast polling for "live" feel
  })

  const handleRefresh = () => {
    refetchSummary()
  }

  const formatViolationType = (type: string) => {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Operational Dashboard</h1>
          <p className="page-subtitle">Real-time traffic safety monitoring & violation analytics</p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-secondary btn-sm" onClick={handleRefresh}>
            <RefreshCw size={14} style={{ marginRight: 4 }} /> Refresh
          </button>
          <Link to="/analyze" className="btn btn-primary btn-sm">
            <Camera size={14} style={{ marginRight: 4 }} /> New Analysis <ArrowRight size={14} />
          </Link>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="stat-grid">
        <div className="stat-card" style={{ '--card-accent': 'var(--amber)' } as React.CSSProperties}>
          <div className="stat-label">Total Violations</div>
          <div className="stat-value">{summaryLoading ? '...' : summary?.total_violations ?? 0}</div>
          <div className="stat-icon"><ShieldAlert size={24} /></div>
          <div className="stat-change">
            <TrendingUp size={12} /> Active Monitoring
          </div>
        </div>

        <div className="stat-card" style={{ '--card-accent': 'var(--red)' } as React.CSSProperties}>
          <div className="stat-label">Today's Incidents</div>
          <div className="stat-value">{summaryLoading ? '...' : summary?.today_violations ?? 0}</div>
          <div className="stat-icon"><Activity size={24} /></div>
          <div className="stat-change" style={{ color: 'var(--red-light)' }}>
            <AlertTriangle size={12} /> Real-time feed
          </div>
        </div>

        <div className="stat-card" style={{ '--card-accent': 'var(--blue)' } as React.CSSProperties}>
          <div className="stat-label">Pending Review</div>
          <div className="stat-value">{summaryLoading ? '...' : summary?.pending_review ?? 0}</div>
          <div className="stat-icon"><Clock size={24} /></div>
          <div className="stat-change" style={{ color: 'var(--blue-light)' }}>
            Needs Officer Verification
          </div>
        </div>

        <div className="stat-card" style={{ '--card-accent': 'var(--green)' } as React.CSSProperties}>
          <div className="stat-label">Avg Confidence</div>
          <div className="stat-value">
            {summaryLoading ? '...' : `${((summary?.avg_confidence ?? 0.82) * 100).toFixed(0)}%`}
          </div>
          <div className="stat-icon"><CheckCircle2 size={24} /></div>
          <div className="stat-change" style={{ color: 'var(--green-light)' }}>
            System accuracy
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 28 }}>
        {/* Trend Area Chart */}
        <div className="chart-container">
          <h3 className="chart-title">Violation Trends (Last 7 Days)</h3>
          <div style={{ height: 260 }}>
            {trendsLoading ? (
              <div className="empty-state">Loading trend data...</div>
            ) : trends && trends.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorViolations" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--amber)" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="var(--amber)" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={11} />
                  <YAxis stroke="var(--text-muted)" fontSize={11} />
                  <Tooltip 
                    contentStyle={{ 
                      background: 'var(--bg-surface)', 
                      borderColor: 'var(--border)', 
                      borderRadius: 8,
                      color: 'var(--text-primary)'
                    }} 
                  />
                  <Area type="monotone" dataKey="count" stroke="var(--amber)" strokeWidth={2} fillOpacity={1} fill="url(#colorViolations)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">No trend data available</div>
            )}
          </div>
        </div>

        {/* Categories Bar Chart */}
        <div className="chart-container">
          <h3 className="chart-title">Distribution by Type</h3>
          <div style={{ height: 260 }}>
            {summaryLoading ? (
              <div className="empty-state">Loading distribution data...</div>
            ) : summary?.top_violations && summary.top_violations.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={summary.top_violations} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis 
                    dataKey="violation_type" 
                    stroke="var(--text-muted)" 
                    fontSize={10} 
                    tickFormatter={formatViolationType}
                  />
                  <YAxis stroke="var(--text-muted)" fontSize={11} />
                  <Tooltip
                    contentStyle={{ 
                      background: 'var(--bg-surface)', 
                      borderColor: 'var(--border)', 
                      borderRadius: 8,
                      color: 'var(--text-primary)'
                    }}
                    formatter={(value) => [value, 'Incidents']}
                    labelFormatter={formatViolationType}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {summary.top_violations.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">No violation data available</div>
            )}
          </div>
        </div>
      </div>

      {/* Live Feed Panel */}
      <div className="card glass">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="status-dot"></span> Live Incident Feed
          </h3>
          <Link to="/violations" className="btn btn-secondary btn-sm">
            View All Records
          </Link>
        </div>

        {violationsLoading ? (
          <div className="empty-state">Loading live feed...</div>
        ) : recentViolations?.items && recentViolations.items.length > 0 ? (
          <div className="violation-list">
            {recentViolations.items.map((v) => (
              <motion.div 
                key={v.id} 
                className="violation-item"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
              >
                <div 
                  className="violation-dot" 
                  style={{ 
                    background: v.violation_type === 'helmet_non_compliance' ? 'var(--v-helmet)' :
                                v.violation_type === 'triple_riding' ? 'var(--v-triple)' :
                                v.violation_type === 'seatbelt_non_compliance' ? 'var(--v-seatbelt)' :
                                v.violation_type === 'wrong_side_driving' ? 'var(--v-wrongside)' :
                                v.violation_type === 'stop_line_violation' ? 'var(--v-stopline)' :
                                v.violation_type === 'red_light_violation' ? 'var(--v-redlight)' :
                                'var(--v-parking)'
                  }} 
                />
                <div>
                  <div className="violation-name">{formatViolationType(v.violation_type)}</div>
                  <div className="violation-desc">
                    Vehicle: <strong style={{ color: 'var(--text-primary)' }}>{v.vehicle_type}</strong>
                    {v.license_plate && (
                      <>
                        {' • '} Plate: <strong style={{ color: 'var(--amber)' }}>{v.license_plate}</strong>
                      </>
                    )}
                    {' • '} {v.location || 'Camera 01'} • {new Date(v.timestamp).toLocaleTimeString()}
                  </div>
                </div>

                <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 15 }}>
                  <span className={`badge ${
                    v.status === 'approved' ? 'badge-green' :
                    v.status === 'dismissed' ? 'badge-gray' : 'badge-amber'
                  }`}>
                    {v.status}
                  </span>
                  
                  <div className="confidence-bar">
                    <div className="conf-value">{Math.round(v.confidence * 100)}%</div>
                    <div className="conf-label">Confidence</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">🛡️</div>
            <div className="empty-state-title">No recent violations detected</div>
            <p>Traffic rules are currently being respected.</p>
          </div>
        )}
      </div>
    </div>
  )
}
