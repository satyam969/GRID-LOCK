import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, 
  Filter, 
  ChevronLeft, 
  ChevronRight, 
  Check, 
  X, 
  Eye, 
  Calendar,
  MapPin,
  Clock,
  Car,
  FileSpreadsheet,
  FileImage,
  ExternalLink
} from 'lucide-react'
import { fetchViolations, updateViolation, downloadCSV, ViolationRecord } from '../api'

const VIOLATION_TYPES = [
  { value: '', label: 'All Violations' },
  { value: 'helmet_non_compliance', label: 'Helmet Non-compliance' },
  { value: 'seatbelt_non_compliance', label: 'Seatbelt Non-compliance' },
  { value: 'triple_riding', label: 'Triple Riding' },
  { value: 'wrong_side_driving', label: 'Wrong-side Driving' },
  { value: 'stop_line_violation', label: 'Stop-line Violation' },
  { value: 'red_light_violation', label: 'Red-light Violation' },
  { value: 'illegal_parking', label: 'Illegal Parking' },
]

const STATUS_OPTIONS = [
  { value: '', label: 'All Statuses' },
  { value: 'pending', label: 'Pending Review' },
  { value: 'approved', label: 'Approved' },
  { value: 'dismissed', label: 'Dismissed' },
]

export default function Violations() {
  const queryClient = useQueryClient()
  const [page, setPage] = useState(1)
  const [pageSize] = useState(10)
  
  // Filter states
  const [violationType, setViolationType] = useState('')
  const [status, setStatus] = useState('')
  const [plateQuery, setPlateQuery] = useState('')
  const [searchPlate, setSearchPlate] = useState('')

  // Detail Modal / Drawer state
  const [selectedViolation, setSelectedViolation] = useState<ViolationRecord | null>(null)

  // Fetching
  const { data, isLoading, isPlaceholderData } = useQuery({
    queryKey: ['violations', page, violationType, status, searchPlate],
    queryFn: () => fetchViolations({
      page,
      page_size: pageSize,
      violation_type: violationType || undefined,
      status: status || undefined,
      license_plate: searchPlate || undefined,
    }),
    placeholderData: (prev) => prev,
  })

  // Mutation for updating status
  const updateStatusMutation = useMutation({
    mutationFn: ({ id, newStatus }: { id: string; newStatus: string }) => 
      updateViolation(id, { status: newStatus }),
    onSuccess: (updatedRecord) => {
      // Refresh current page query
      queryClient.invalidateQueries({ queryKey: ['violations'] })
      queryClient.invalidateQueries({ queryKey: ['summary'] })
      // Update selected detail overlay if it is open
      if (selectedViolation && selectedViolation.id === updatedRecord.id) {
        setSelectedViolation(updatedRecord)
      }
    }
  })

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setPage(1)
    setSearchPlate(plateQuery)
  }

  const handleResetFilters = () => {
    setViolationType('')
    setStatus('')
    setPlateQuery('')
    setSearchPlate('')
    setPage(1)
  }

  const handleUpdateStatus = (id: string, newStatus: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation()
    updateStatusMutation.mutate({ id, newStatus })
  }

  const formatViolationType = (type: string) => {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  }

  const getSeverityBadge = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical':
      case 'high':
        return 'badge-red'
      case 'medium':
        return 'badge-amber'
      default:
        return 'badge-blue'
    }
  }

  const getStatusBadge = (statusVal: string) => {
    switch (statusVal?.toLowerCase()) {
      case 'approved':
        return 'badge-green'
      case 'dismissed':
        return 'badge-gray'
      default:
        return 'badge-amber'
    }
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Violation Records</h1>
          <p className="page-subtitle">Verify, search, and export documented traffic offenses</p>
        </div>
        <div>
          <button className="btn btn-secondary btn-sm" onClick={() => downloadCSV()}>
            <FileSpreadsheet size={14} style={{ marginRight: 4 }} /> Export CSV
          </button>
        </div>
      </div>

      {/* Filters Card */}
      <div className="card glass" style={{ marginBottom: 20 }}>
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end' }}>
          
          <div style={{ flex: 2, minWidth: 200, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Search License Plate</label>
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                placeholder="e.g., MH12AB1234"
                className="input"
                value={plateQuery}
                onChange={e => setPlateQuery(e.target.value)}
                style={{ paddingLeft: 34 }}
              />
              <Search size={14} style={{ position: 'absolute', left: 12, top: 12, color: 'var(--text-muted)' }} />
            </div>
          </div>

          <div style={{ flex: 1, minWidth: 150, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Violation Category</label>
            <select
              className="input"
              value={violationType}
              onChange={e => { setViolationType(e.target.value); setPage(1); }}
            >
              {VIOLATION_TYPES.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div style={{ flex: 1, minWidth: 150, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <label style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Status</label>
            <select
              className="input"
              value={status}
              onChange={e => { setStatus(e.target.value); setPage(1); }}
            >
              {STATUS_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button type="submit" className="btn btn-primary btn-sm" style={{ height: 38 }}>
              Filter
            </button>
            {(violationType || status || searchPlate) && (
              <button type="button" className="btn btn-secondary btn-sm" onClick={handleResetFilters} style={{ height: 38 }}>
                Reset
              </button>
            )}
          </div>
        </form>
      </div>

      {/* Table Container */}
      <div className="table-wrap">
        {isLoading ? (
          <div className="empty-state">
            <div className="spinner" style={{ marginBottom: 12 }} />
            <p>Loading database records...</p>
          </div>
        ) : data?.items && data.items.length > 0 ? (
          <>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Violation Type</th>
                  <th>Vehicle</th>
                  <th>License Plate</th>
                  <th>Confidence</th>
                  <th>Severity</th>
                  <th>Status</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((record) => (
                  <tr 
                    key={record.id} 
                    onClick={() => setSelectedViolation(record)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>
                      <div style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                        {new Date(record.timestamp).toLocaleDateString()}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
                        {new Date(record.timestamp).toLocaleTimeString()}
                      </div>
                    </td>
                    <td>
                      <strong style={{ color: 'var(--text-primary)' }}>
                        {formatViolationType(record.violation_type)}
                      </strong>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
                        {record.location || 'Camera 01'}
                      </div>
                    </td>
                    <td style={{ textTransform: 'capitalize' }}>{record.vehicle_type}</td>
                    <td>
                      {record.license_plate ? (
                        <span className="plate-display" style={{ fontSize: '0.75rem', padding: '2px 8px', borderWidth: 1 }}>
                          {record.license_plate}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>None</span>
                      )}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>
                      {Math.round(record.confidence * 100)}%
                    </td>
                    <td>
                      <span className={`badge ${getSeverityBadge(record.severity)}`}>
                        {record.severity}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${getStatusBadge(record.status)}`}>
                        {record.status}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }} onClick={e => e.stopPropagation()}>
                      <div style={{ display: 'inline-flex', gap: 6 }}>
                        <button 
                          className="btn btn-secondary btn-sm" 
                          style={{ padding: 6 }} 
                          title="View Details"
                          onClick={() => setSelectedViolation(record)}
                        >
                          <Eye size={13} />
                        </button>
                        
                        {record.status === 'pending' && (
                          <>
                            <button 
                              className="btn btn-secondary btn-sm" 
                              style={{ padding: 6, color: 'var(--green-light)', borderColor: 'rgba(16,185,129,0.2)' }}
                              title="Approve Violation"
                              onClick={(e) => handleUpdateStatus(record.id, 'approved', e)}
                              disabled={updateStatusMutation.isPending}
                            >
                              <Check size={13} />
                            </button>
                            <button 
                              className="btn btn-secondary btn-sm" 
                              style={{ padding: 6, color: 'var(--red-light)', borderColor: 'rgba(239,68,68,0.2)' }}
                              title="Dismiss Incident"
                              onClick={(e) => handleUpdateStatus(record.id, 'dismissed', e)}
                              disabled={updateStatusMutation.isPending}
                            >
                              <X size={13} />
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination footer */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              padding: '12px 16px', 
              borderTop: '1px solid var(--border)',
              background: 'rgba(0,0,0,0.1)'
            }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Showing <strong>{((page - 1) * pageSize) + 1}</strong> - <strong>{Math.min(page * pageSize, data.total)}</strong> of <strong>{data.total}</strong> records
              </span>

              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1 || isPlaceholderData}
                >
                  <ChevronLeft size={14} /> Prev
                </button>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setPage(p => Math.min(data.pages, p + 1))}
                  disabled={page === data.pages || isPlaceholderData}
                >
                  Next <ChevronRight size={14} />
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">📋</div>
            <div className="empty-state-title">No matching records found</div>
            <p>Try clearing your queries or selecting different filter options.</p>
          </div>
        )}
      </div>

      {/* Slide-over Detail Drawer */}
      <AnimatePresence>
        {selectedViolation && (
          <>
            {/* Backdrop */}
            <motion.div 
              style={{
                position: 'fixed',
                top: 0, left: 0, right: 0, bottom: 0,
                background: 'rgba(0, 0, 0, 0.6)',
                backdropFilter: 'blur(4px)',
                zIndex: 200
              }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedViolation(null)}
            />

            {/* Side Drawer */}
            <motion.div
              style={{
                position: 'fixed',
                top: 0, right: 0, bottom: 0,
                width: 'min(550px, 90vw)',
                background: 'var(--bg-surface)',
                borderLeft: '1px solid var(--border)',
                boxShadow: '-8px 0 32px rgba(0,0,0,0.5)',
                zIndex: 201,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column'
              }}
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'tween', duration: 0.25 }}
            >
              {/* Header */}
              <div style={{ 
                padding: '20px 24px', 
                borderBottom: '1px solid var(--border)', 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center' 
              }}>
                <div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Record: #{selectedViolation.id.substring(0, 8)}
                  </div>
                  <h2 style={{ fontSize: '1.25rem', marginTop: 4 }}>
                    {formatViolationType(selectedViolation.violation_type)}
                  </h2>
                </div>
                <button 
                  className="btn btn-secondary btn-sm" 
                  style={{ padding: 6, borderRadius: '50%' }}
                  onClick={() => setSelectedViolation(null)}
                >
                  <X size={16} />
                </button>
              </div>

              {/* Body */}
              <div style={{ padding: 24, flex: 1, display: 'flex', flexDirection: 'column', gap: 20 }}>
                {/* Images */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>EVIDENCE DOCUMENTATION</div>
                  <div className="image-box" style={{ borderColor: 'var(--amber-dark)' }}>
                    <div className="image-box-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>AI Model Bounding Box Overlays</span>
                      {selectedViolation.annotated_image_path && (
                        <a href={selectedViolation.annotated_image_path} target="_blank" rel="noreferrer" style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: 3 }}>
                          View Raw <ExternalLink size={10} />
                        </a>
                      )}
                    </div>
                    <img 
                      src={selectedViolation.annotated_image_path || '/placeholder.jpg'} 
                      alt="Annotated violation evidence" 
                      style={{ height: 260 }}
                    />
                  </div>
                </div>

                {/* Status bar actions */}
                <div className="card" style={{ background: 'rgba(255,255,255,0.02)', padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>INCIDENT MODERATION</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                      <span className={`badge ${getStatusBadge(selectedViolation.status)}`}>
                        {selectedViolation.status}
                      </span>
                    </div>
                  </div>
                  
                  {selectedViolation.status === 'pending' && (
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button 
                        className="btn btn-secondary btn-sm"
                        style={{ color: 'var(--green-light)', borderColor: 'rgba(16,185,129,0.2)', background: 'rgba(16,185,129,0.05)' }}
                        onClick={() => handleUpdateStatus(selectedViolation.id, 'approved')}
                        disabled={updateStatusMutation.isPending}
                      >
                        <Check size={14} style={{ marginRight: 4 }} /> Approve
                      </button>
                      <button 
                        className="btn btn-secondary btn-sm"
                        style={{ color: 'var(--red-light)', borderColor: 'rgba(239,68,68,0.2)', background: 'rgba(239,68,68,0.05)' }}
                        onClick={() => handleUpdateStatus(selectedViolation.id, 'dismissed')}
                        disabled={updateStatusMutation.isPending}
                      >
                        <X size={14} style={{ marginRight: 4 }} /> Dismiss
                      </button>
                    </div>
                  )}
                </div>

                {/* Metadata */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>METADATA & TELEMETRY</div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    
                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: 12, borderRadius: 6, border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>CAMERA ID</div>
                      <div style={{ fontSize: '0.875rem', fontWeight: 600, marginTop: 2, display: 'flex', alignItems: 'center', gap: 4 }}>
                        <MapPin size={12} className="v-stopline" /> {selectedViolation.camera_id || 'Camera 01'}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: 12, borderRadius: 6, border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>VEHICLE CLASSIFICATION</div>
                      <div style={{ fontSize: '0.875rem', fontWeight: 600, marginTop: 2, display: 'flex', alignItems: 'center', gap: 4, textTransform: 'capitalize' }}>
                        <Car size={12} className="v-wrongside" /> {selectedViolation.vehicle_type}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: 12, borderRadius: 6, border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>TIMESTAMP</div>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, marginTop: 2, display: 'flex', alignItems: 'center', gap: 4 }}>
                        <Clock size={12} /> {new Date(selectedViolation.timestamp).toLocaleString()}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: 12, borderRadius: 6, border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>DETECTION CONFIDENCE</div>
                      <div style={{ fontSize: '0.875rem', fontWeight: 600, marginTop: 2, fontFamily: 'var(--font-mono)', color: 'var(--amber)' }}>
                        {Math.round(selectedViolation.confidence * 100)}%
                      </div>
                    </div>

                  </div>

                  {/* License Plate Special Render */}
                  {selectedViolation.license_plate && (
                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: 16, borderRadius: 6, border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>EXTRACTED PLATE NUMBER</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
                          OCR Match ({(selectedViolation.plate_confidence ? Math.round(selectedViolation.plate_confidence * 100) : 85)}% Match)
                        </div>
                      </div>
                      <div className="plate-display">{selectedViolation.license_plate}</div>
                    </div>
                  )}

                  {selectedViolation.person_count && selectedViolation.person_count > 1 && (
                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: 12, borderRadius: 6, border: '1px solid var(--border)', fontSize: '0.8rem' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Rider / Occupant Counting: </span> 
                      <strong>{selectedViolation.person_count} persons</strong> detected on vehicle.
                    </div>
                  )}

                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
