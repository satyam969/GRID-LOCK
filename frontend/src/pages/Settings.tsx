import React, { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchSettings, updateSettings } from '../api'
import { Settings as SettingsIcon, ShieldAlert, Cpu, Mail, Save, AlertTriangle } from 'lucide-react'

export default function Settings() {
  const queryClient = useQueryClient()
  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
  })

  const [formData, setFormData] = useState<any>(null)

  useEffect(() => {
    if (settings) setFormData(settings)
  }, [settings])

  const updateMutation = useMutation({
    mutationFn: (data: any) => updateSettings(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      alert('Settings updated successfully!')
    }
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    updateMutation.mutate(formData)
  }

  if (isLoading || !formData) return <div className="spinner" style={{ margin: 'auto' }} />

  return (
    <div className="fade-in" style={{ maxWidth: 800, margin: '0 auto' }}>
      <div className="page-header" style={{ marginBottom: 30 }}>
        <div>
          <h1 className="page-title">System Settings</h1>
          <p className="page-subtitle">Configure AI confidence thresholds and application behavior</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
        
        {/* Detection Thresholds */}
        <div className="card glass">
          <h2 style={{ fontSize: '1.2rem', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Cpu size={20} className="v-helmet" /> AI Confidence Thresholds
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 24 }}>
            Set the minimum confidence score required for the AI to flag a violation. Lower values increase false positives, while higher values may miss actual violations.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <label style={{ fontWeight: 600 }}>General Detection</label>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--amber)' }}>{formData.conf_general.toFixed(2)}</span>
              </div>
              <input type="range" min="0.1" max="0.9" step="0.05" value={formData.conf_general} onChange={e => setFormData({...formData, conf_general: parseFloat(e.target.value)})} style={{ width: '100%' }} />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <label style={{ fontWeight: 600 }}>Helmet Detection</label>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--amber)' }}>{formData.conf_helmet.toFixed(2)}</span>
              </div>
              <input type="range" min="0.1" max="0.9" step="0.05" value={formData.conf_helmet} onChange={e => setFormData({...formData, conf_helmet: parseFloat(e.target.value)})} style={{ width: '100%' }} />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <label style={{ fontWeight: 600 }}>Seatbelt Detection</label>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--amber)' }}>{formData.conf_seatbelt.toFixed(2)}</span>
              </div>
              <input type="range" min="0.1" max="0.9" step="0.05" value={formData.conf_seatbelt} onChange={e => setFormData({...formData, conf_seatbelt: parseFloat(e.target.value)})} style={{ width: '100%' }} />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <label style={{ fontWeight: 600 }}>License Plate OCR</label>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--amber)' }}>{formData.conf_plate.toFixed(2)}</span>
              </div>
              <input type="range" min="0.1" max="0.9" step="0.05" value={formData.conf_plate} onChange={e => setFormData({...formData, conf_plate: parseFloat(e.target.value)})} style={{ width: '100%' }} />
            </div>
          </div>
        </div>

        {/* Global Behavior */}
        <div className="card glass">
          <h2 style={{ fontSize: '1.2rem', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
            <SettingsIcon size={20} className="v-seatbelt" /> System Behavior
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer' }}>
              <input type="checkbox" checked={formData.enable_clahe} onChange={e => setFormData({...formData, enable_clahe: e.target.checked})} style={{ width: 18, height: 18 }} />
              <div>
                <div style={{ fontWeight: 600 }}>Enable CLAHE Preprocessing</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Enhances image contrast for night-time captures, but adds ~15ms processing overhead.</div>
              </div>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer' }}>
              <input type="checkbox" checked={formData.auto_generate_challans} onChange={e => setFormData({...formData, auto_generate_challans: e.target.checked})} style={{ width: 18, height: 18 }} />
              <div>
                <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>Auto-Generate Challans <AlertTriangle size={14} color="var(--red-light)" /></div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Automatically issue challans for HIGH severity violations without officer approval.</div>
              </div>
            </label>
          </div>
        </div>

        {/* Notifications */}
        <div className="card glass">
          <h2 style={{ fontSize: '1.2rem', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Mail size={20} className="v-wrongside" /> Notifications
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer' }}>
              <input type="checkbox" checked={formData.email_alerts} onChange={e => setFormData({...formData, email_alerts: e.target.checked})} style={{ width: 18, height: 18 }} />
              <div>
                <div style={{ fontWeight: 600 }}>Enable Email Alerts</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Send daily digests and repeat offender alerts.</div>
              </div>
            </label>

            <div style={{ marginTop: 8 }}>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>Admin Contact Email</label>
              <input type="email" className="input" value={formData.admin_email} onChange={e => setFormData({...formData, admin_email: e.target.value})} style={{ width: '100%', maxWidth: 400, marginTop: 4 }} />
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10 }}>
          <button type="submit" className="btn btn-primary" style={{ padding: '12px 24px', fontSize: '1rem' }} disabled={updateMutation.isPending}>
            <Save size={18} style={{ marginRight: 8 }} /> Save Configurations
          </button>
        </div>
      </form>
    </div>
  )
}
