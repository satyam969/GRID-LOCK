import { Routes, Route, NavLink, useLocation, Navigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from './api'

import Dashboard from './pages/Dashboard'
import Analyze from './pages/Analyze'
import Violations from './pages/Violations'
import VehicleLookup from './pages/VehicleLookup'
import ChallanManagement from './pages/ChallanManagement'
import CameraManagement from './pages/CameraManagement'
import Settings from './pages/Settings'
import Reports from './pages/Reports'
import Analytics from './pages/Analytics'

const NAV = [
  { path: '/',          icon: '⚡', label: 'Dashboard' },
  { path: '/analyze',   icon: '🔍', label: 'Analyze' },
  { path: '/violations',icon: '📋', label: 'Violations' },
  { path: '/vehicles',  icon: '🚘', label: 'Vehicles' },
  { path: '/challans',  icon: '📄', label: 'Challans' },
  { path: '/cameras',   icon: '📹', label: 'Cameras' },
  { path: '/analytics', icon: '📊', label: 'Analytics' },
  { path: '/reports',   icon: '📥', label: 'Reports' },
  { path: '/settings',  icon: '⚙️', label: 'Settings' },
]

function Sidebar() {
  const location = useLocation()
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 15_000,
  })

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-mark">
          <div className="logo-icon">🚦</div>
          <div>
            <div className="logo-text">TrafficGuard</div>
            <div className="logo-sub">AI Detection System</div>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-label">Navigation</div>
        {NAV.map(({ path, icon, label }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <span className="nav-icon">{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="system-status">
          <div className="status-dot" style={{
            background: health?.status === 'ok' ? 'var(--green)' : 'var(--red)',
            boxShadow: `0 0 6px ${health?.status === 'ok' ? 'var(--green)' : 'var(--red)'}`,
          }} />
          <span>{health?.status === 'ok' ? 'System Online' : 'Connecting...'}</span>
        </div>
        {health && (
          <div style={{ marginTop: 8, fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {Object.entries(health.models_loaded || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
                <span>{k}</span>
                <span style={{ color: v ? 'var(--green)' : 'var(--red)' }}>{v ? '✓' : '✗'}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}

export default function App() {
  const location = useLocation()

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            transition={{ duration: 0.2 }}
          >
            <Routes>
              <Route path="/"           element={<Dashboard />} />
              <Route path="/analyze"    element={<Analyze />} />
              <Route path="/violations" element={<Violations />} />
              <Route path="/vehicles"   element={<VehicleLookup />} />
              <Route path="/challans"   element={<ChallanManagement />} />
              <Route path="/cameras"    element={<CameraManagement />} />
              <Route path="/analytics"  element={<Analytics />} />
              <Route path="/reports"    element={<Reports />} />
              <Route path="/settings"   element={<Settings />} />
              <Route path="*"           element={<Navigate to="/" replace />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
