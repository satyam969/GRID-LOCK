import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchCameras, createCamera, updateCamera, deleteCamera } from '../api'
import { Video, Plus, MapPin, Activity, Settings2, Trash2 } from 'lucide-react'

export default function CameraManagement() {
  const queryClient = useQueryClient()
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({ name: '', location: '', zone: '', direction: '', latitude: '', longitude: '' })

  const { data: cameras, isLoading } = useQuery({
    queryKey: ['cameras'],
    queryFn: fetchCameras,
  })

  const createMutation = useMutation({
    mutationFn: (data: any) => createCamera(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cameras'] })
      setShowForm(false)
      setFormData({ name: '', location: '', zone: '', direction: '', latitude: '', longitude: '' })
    }
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteCamera(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cameras'] })
    }
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    // Parse coordinates
    const payload: any = { ...formData }
    if (payload.latitude === '') delete payload.latitude; else payload.latitude = parseFloat(payload.latitude);
    if (payload.longitude === '') delete payload.longitude; else payload.longitude = parseFloat(payload.longitude);
    
    createMutation.mutate(payload)
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Camera Management</h1>
          <p className="page-subtitle">Configure IoT camera nodes and traffic zones</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
          <Plus size={16} /> Add Camera
        </button>
      </div>

      {showForm && (
        <div className="card glass" style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: '1.1rem', marginBottom: 16 }}>Register New Camera</h3>
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 200px' }}>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Camera Name</label>
              <input required className="input" style={{ width: '100%', marginTop: 4 }} value={formData.name} onChange={e => setFormData({...formData, name: e.target.value})} placeholder="e.g. CAM-NORTH-01" />
            </div>
            <div style={{ flex: '1 1 200px' }}>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Location</label>
              <input required className="input" style={{ width: '100%', marginTop: 4 }} value={formData.location} onChange={e => setFormData({...formData, location: e.target.value})} placeholder="e.g. MG Road Intersection" />
            </div>
            <div style={{ flex: '1 1 200px' }}>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Enforcement Zone</label>
              <select required className="input" style={{ width: '100%', marginTop: 4 }} value={formData.zone} onChange={e => setFormData({...formData, zone: e.target.value})}>
                <option value="">Select Zone</option>
                <option value="North">North Zone</option>
                <option value="South">South Zone</option>
                <option value="East">East Zone</option>
                <option value="West">West Zone</option>
                <option value="Central">Central Zone</option>
              </select>
            </div>
            <div style={{ flex: '1 1 200px' }}>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Direction</label>
              <select required className="input" style={{ width: '100%', marginTop: 4 }} value={formData.direction} onChange={e => setFormData({...formData, direction: e.target.value})}>
                <option value="">Select Direction</option>
                <option value="Inbound">Inbound</option>
                <option value="Outbound">Outbound</option>
              </select>
            </div>
            <div style={{ flex: '1 1 100px' }}>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Latitude (Optional)</label>
              <input type="number" step="any" className="input" style={{ width: '100%', marginTop: 4 }} value={formData.latitude} onChange={e => setFormData({...formData, latitude: e.target.value})} placeholder="e.g. 28.7041" />
            </div>
            <div style={{ flex: '1 1 100px' }}>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Longitude (Optional)</label>
              <input type="number" step="any" className="input" style={{ width: '100%', marginTop: 4 }} value={formData.longitude} onChange={e => setFormData({...formData, longitude: e.target.value})} placeholder="e.g. 77.1025" />
            </div>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'flex-end', gap: 12, marginTop: 8 }}>
              <button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button>
              <button type="submit" className="btn btn-primary" disabled={createMutation.isPending}>Save Camera</button>
            </div>
          </form>
        </div>
      )}

      <div className="grid-2">
        {isLoading ? (
          <div className="spinner" />
        ) : cameras?.length > 0 ? (
          cameras.map((cam: any) => (
            <div key={cam.id} className="card glass" style={{ display: 'flex', gap: 20 }}>
              <div style={{ padding: 16, background: 'rgba(59, 130, 246, 0.1)', borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Video size={32} color="var(--blue-light)" />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>{cam.name}</h3>
                      <span className={`badge ${cam.status === 'active' ? 'badge-green' : 'badge-red'}`}>{cam.status}</span>
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <MapPin size={14} /> {cam.location}
                    </div>
                  </div>
                  <button 
                    className="btn btn-secondary btn-sm" 
                    style={{ padding: 6, borderColor: 'var(--red)', color: 'var(--red)' }}
                    onClick={() => {
                      if (confirm('Decommission this camera?')) deleteMutation.mutate(cam.id)
                    }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
                
                <div style={{ display: 'flex', gap: 16, marginTop: 16 }}>
                  <div style={{ flex: 1, background: 'rgba(255,255,255,0.03)', padding: '8px 12px', borderRadius: 8 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Zone</div>
                    <div style={{ fontWeight: 500 }}>{cam.zone}</div>
                  </div>
                  <div style={{ flex: 1, background: 'rgba(255,255,255,0.03)', padding: '8px 12px', borderRadius: 8 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Direction</div>
                    <div style={{ fontWeight: 500 }}>{cam.direction}</div>
                  </div>
                  <div style={{ flex: 1, background: 'rgba(255,255,255,0.03)', padding: '8px 12px', borderRadius: 8 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Status</div>
                    <div style={{ fontWeight: 500, color: 'var(--green-light)' }}>Online</div>
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="empty-state" style={{ gridColumn: '1 / -1' }}>
            <div className="empty-state-icon">📹</div>
            <div className="empty-state-title">No Cameras Registered</div>
            <p>Click "Add Camera" to register IoT nodes.</p>
          </div>
        )}
      </div>
    </div>
  )
}
