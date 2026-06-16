import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { searchVehicles, fetchVehicle, fetchVehicleViolations } from '../api'
import { Search, AlertTriangle, Car, Calendar, ShieldAlert } from 'lucide-react'

export default function VehicleLookup() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedVehicleId, setSelectedVehicleId] = useState<number | null>(null)

  // Search autocomplete query
  const { data: searchResults, isLoading: isSearching } = useQuery({
    queryKey: ['vehicleSearch', searchQuery],
    queryFn: () => searchVehicles(searchQuery),
    enabled: searchQuery.length >= 2,
  })

  // Full vehicle profile
  const { data: vehicle, isLoading: isLoadingVehicle } = useQuery({
    queryKey: ['vehicle', selectedVehicleId],
    queryFn: () => fetchVehicle(selectedVehicleId!),
    enabled: !!selectedVehicleId,
  })

  // Vehicle violations history
  const { data: violations, isLoading: isLoadingViolations } = useQuery({
    queryKey: ['vehicleViolations', selectedVehicleId],
    queryFn: () => fetchVehicleViolations(selectedVehicleId!),
    enabled: !!selectedVehicleId,
  })

  const formatViolationType = (type: string) => {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Vehicle Lookup</h1>
          <p className="page-subtitle">Search by license plate to view full violation history and repeat offender status</p>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
        
        {/* Left Col: Search & Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="card glass">
            <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 16 }}>Search License Plate</h3>
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                className="input"
                placeholder="e.g. WB06J2431"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ width: '100%', paddingLeft: 40, fontSize: '1.1rem', letterSpacing: '2px', textTransform: 'uppercase' }}
              />
              <Search style={{ position: 'absolute', left: 14, top: 10, color: 'var(--text-muted)' }} />
            </div>

            <div style={{ marginTop: 20 }}>
              {isSearching && <div className="spinner" style={{ margin: '0 auto' }} />}
              {!isSearching && searchResults && searchResults.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {searchResults.map((v: any) => (
                    <div 
                      key={v.id} 
                      onClick={() => setSelectedVehicleId(v.id)}
                      style={{
                        padding: '12px 16px',
                        background: selectedVehicleId === v.id ? 'rgba(59, 130, 246, 0.1)' : 'rgba(255,255,255,0.03)',
                        border: `1px solid ${selectedVehicleId === v.id ? 'var(--blue)' : 'var(--border)'}`,
                        borderRadius: 8,
                        cursor: 'pointer',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}
                    >
                      <div>
                        <div className="plate-display" style={{ fontSize: '0.8rem', padding: '2px 6px', borderWidth: 1 }}>{v.plate_number}</div>
                      </div>
                      <div style={{ display: 'flex', gap: 8 }}>
                        {v.is_repeat_offender && <span className="badge badge-red" title="Repeat Offender">⚠️</span>}
                        <span className="badge badge-blue">{v.total_violations} offenses</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {!isSearching && searchResults && searchResults.length === 0 && searchQuery.length >= 2 && (
                <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', padding: 20 }}>
                  No vehicles found matching "{searchQuery}"
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Col: Vehicle Profile */}
        <div>
          {!selectedVehicleId ? (
            <div className="empty-state" style={{ height: '100%' }}>
              <div className="empty-state-icon">🚘</div>
              <div className="empty-state-title">Select a Vehicle</div>
              <p>Search and select a vehicle to view its complete history.</p>
            </div>
          ) : isLoadingVehicle ? (
            <div className="empty-state" style={{ height: '100%' }}>
              <div className="spinner" />
            </div>
          ) : vehicle ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              
              {/* Profile Card */}
              <div className="card glass" style={{ position: 'relative', overflow: 'hidden' }}>
                {vehicle.is_repeat_offender && (
                  <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 4, background: 'var(--red)' }} />
                )}
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <div className="plate-display" style={{ fontSize: '1.5rem' }}>{vehicle.plate_number}</div>
                    <div style={{ display: 'flex', gap: 16, marginTop: 12, color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 6, textTransform: 'capitalize' }}>
                        <Car size={16}/> {vehicle.vehicle_type}
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <Calendar size={16}/> First seen: {new Date(vehicle.first_seen).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 700, color: vehicle.is_repeat_offender ? 'var(--red-light)' : 'var(--blue-light)', lineHeight: 1 }}>
                      {vehicle.total_violations}
                    </div>
                    <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, marginTop: 4 }}>
                      Total Violations
                    </div>
                  </div>
                </div>

                {vehicle.is_repeat_offender && (
                  <div style={{ marginTop: 20, padding: 12, background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: 8, display: 'flex', gap: 12, alignItems: 'center', color: 'var(--red-light)' }}>
                    <ShieldAlert size={24} />
                    <div>
                      <div style={{ fontWeight: 600 }}>Repeat Offender Status Active</div>
                      <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>This vehicle has 3 or more documented violations. Double fine penalties may apply.</div>
                    </div>
                  </div>
                )}
              </div>

              {/* Violations Timeline */}
              <div className="card glass">
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: 16 }}>Violation History</h3>
                
                {isLoadingViolations ? (
                  <div className="spinner" />
                ) : violations && violations.length > 0 ? (
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Violation</th>
                          <th>Confidence</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {violations.map((v: any) => (
                          <tr key={v.id}>
                            <td>
                              <div style={{ fontWeight: 500 }}>{new Date(v.timestamp).toLocaleDateString()}</div>
                              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{new Date(v.timestamp).toLocaleTimeString()}</div>
                            </td>
                            <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                              {formatViolationType(v.violation_type)}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{Math.round(v.confidence * 100)}%</td>
                            <td>
                              <span className={`badge ${v.status === 'challan_issued' ? 'badge-red' : v.status === 'confirmed' ? 'badge-green' : 'badge-blue'}`}>
                                {v.status}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No recorded violations.</div>
                )}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
