import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchChallans, payChallan } from '../api'
import { FileText, IndianRupee, Search, CheckCircle2 } from 'lucide-react'

export default function ChallanManagement() {
  const queryClient = useQueryClient()
  const [statusFilter, setStatusFilter] = useState('')

  const { data: challans, isLoading } = useQuery({
    queryKey: ['challans', statusFilter],
    queryFn: () => fetchChallans(statusFilter || undefined),
  })

  const payMutation = useMutation({
    mutationFn: (id: number) => payChallan(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['challans'] })
    }
  })

  const handlePay = (id: number) => {
    if (confirm('Mark this challan as paid and resolved?')) {
      payMutation.mutate(id)
    }
  }

  // Summary stats
  const totalIssued = challans?.length || 0
  const totalAmount = challans?.reduce((sum: number, c: any) => sum + c.fine_amount, 0) || 0
  const collected = challans?.filter((c: any) => c.payment_status === 'PAID').reduce((sum: number, c: any) => sum + c.fine_amount, 0) || 0
  const pendingAmount = totalAmount - collected

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Challan Management</h1>
          <p className="page-subtitle">Track issued fines and revenue collection</p>
        </div>
      </div>

      {/* Revenue KPI Cards */}
      <div className="grid" style={{ marginBottom: 24, gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        <div className="stat-card" style={{ '--card-accent': 'var(--blue)' } as any}>
          <div className="stat-label">Total Issued</div>
          <div className="stat-value">{totalIssued}</div>
          <div className="stat-icon"><FileText size={24} /></div>
        </div>
        <div className="stat-card" style={{ '--card-accent': 'var(--purple)' } as any}>
          <div className="stat-label">Total Fine Value</div>
          <div className="stat-value">₹{totalAmount.toLocaleString()}</div>
          <div className="stat-icon"><IndianRupee size={24} /></div>
        </div>
        <div className="stat-card" style={{ '--card-accent': 'var(--green)' } as any}>
          <div className="stat-label">Revenue Collected</div>
          <div className="stat-value">₹{collected.toLocaleString()}</div>
          <div className="stat-icon"><CheckCircle2 size={24} /></div>
        </div>
        <div className="stat-card" style={{ '--card-accent': 'var(--red)' } as any}>
          <div className="stat-label">Pending Recovery</div>
          <div className="stat-value">₹{pendingAmount.toLocaleString()}</div>
          <div className="stat-icon"><IndianRupee size={24} /></div>
        </div>
      </div>

      <div className="card glass">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Issued Challans</h3>
          
          <select 
            className="input" 
            style={{ width: 200 }}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="">All Statuses</option>
            <option value="UNPAID">Unpaid</option>
            <option value="PAID">Paid</option>
          </select>
        </div>

        <div className="table-wrap">
          {isLoading ? (
            <div className="spinner" />
          ) : challans && challans.length > 0 ? (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Challan No.</th>
                  <th>Date Issued</th>
                  <th>Vehicle Plate</th>
                  <th>Offense</th>
                  <th>Fine Amount</th>
                  <th>Status</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {challans.map((c: any) => (
                  <tr key={c.id}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{c.challan_number}</td>
                    <td>{new Date(c.created_at).toLocaleDateString()}</td>
                    <td>
                      <span className="plate-display" style={{ fontSize: '0.75rem', padding: '2px 6px', borderWidth: 1 }}>
                        {c.plate_number}
                      </span>
                    </td>
                    <td style={{ textTransform: 'capitalize' }}>{c.violation_type?.replace(/_/g, ' ')}</td>
                    <td style={{ fontWeight: 600 }}>₹{c.fine_amount}</td>
                    <td>
                      <span className={`badge ${c.payment_status === 'PAID' ? 'badge-green' : 'badge-red'}`}>
                        {c.payment_status}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      {c.payment_status === 'UNPAID' ? (
                        <button 
                          className="btn btn-primary btn-sm"
                          onClick={() => handlePay(c.id)}
                          disabled={payMutation.isPending}
                        >
                          Mark Paid
                        </button>
                      ) : (
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                          Paid on {new Date(c.payment_date).toLocaleDateString()}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">📄</div>
              <div className="empty-state-title">No Challans Found</div>
              <p>No challans match the current filters.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
