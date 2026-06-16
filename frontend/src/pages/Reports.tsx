import React, { useState } from 'react'
import { FileDown, Calendar, Filter, PieChart, Receipt } from 'lucide-react'
import { generateReport } from '../api'

export default function Reports() {
  const [reportType, setReportType] = useState('daily_summary')
  const [dateRange, setDateRange] = useState('7days')

  const handleDownload = () => {
    generateReport(reportType)
  }

  return (
    <div className="fade-in" style={{ maxWidth: 900, margin: '0 auto' }}>
      <div className="page-header" style={{ marginBottom: 30 }}>
        <div>
          <h1 className="page-title">Reports Generator</h1>
          <p className="page-subtitle">Export aggregate traffic statistics and revenue data</p>
        </div>
      </div>

      <div className="grid-2">
        
        {/* Generator Form */}
        <div className="card glass" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <h2 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: 8 }}>
            <FileDown size={20} className="v-stopline" /> Export Configuration
          </h2>

          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>Report Type</label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
              <label 
                className={`card ${reportType === 'daily_summary' ? 'active-border' : ''}`}
                style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 16, padding: '16px', border: reportType === 'daily_summary' ? '1px solid var(--amber)' : '1px solid var(--border)' }}
              >
                <input type="radio" name="reportType" value="daily_summary" checked={reportType === 'daily_summary'} onChange={() => setReportType('daily_summary')} style={{ display: 'none' }} />
                <div style={{ background: 'rgba(245, 158, 11, 0.1)', padding: 12, borderRadius: '50%' }}><PieChart size={24} color="var(--amber)" /></div>
                <div>
                  <div style={{ fontWeight: 600 }}>Daily Incident Summary</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Aggregated violations with confidence and status.</div>
                </div>
              </label>

              <label 
                className={`card ${reportType === 'challan_revenue' ? 'active-border' : ''}`}
                style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 16, padding: '16px', border: reportType === 'challan_revenue' ? '1px solid var(--green-light)' : '1px solid var(--border)' }}
              >
                <input type="radio" name="reportType" value="challan_revenue" checked={reportType === 'challan_revenue'} onChange={() => setReportType('challan_revenue')} style={{ display: 'none' }} />
                <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: 12, borderRadius: '50%' }}><Receipt size={24} color="var(--green-light)" /></div>
                <div>
                  <div style={{ fontWeight: 600 }}>Challan Revenue Report</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Issued fines, collected revenue, and outstanding dues.</div>
                </div>
              </label>
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>Time Range</label>
            <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
              <select className="input" style={{ flex: 1 }} value={dateRange} onChange={e => setDateRange(e.target.value)}>
                <option value="today">Today</option>
                <option value="7days">Last 7 Days</option>
                <option value="30days">Last 30 Days</option>
                <option value="ytd">Year to Date</option>
                <option value="custom">Custom Range...</option>
              </select>
            </div>
          </div>

          <button className="btn btn-primary" style={{ marginTop: 'auto', padding: '14px', fontSize: '1rem', display: 'flex', justifyContent: 'center' }} onClick={handleDownload}>
            <FileDown size={18} style={{ marginRight: 8 }} /> Download CSV
          </button>
        </div>

        {/* Info Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card glass" style={{ background: 'rgba(59, 130, 246, 0.05)', borderColor: 'rgba(59, 130, 246, 0.2)' }}>
            <h3 style={{ fontSize: '1rem', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6, color: 'var(--blue-light)' }}>
              <Calendar size={16} /> Automated Scheduling
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              You can configure reports to be generated automatically and emailed to the administrative contacts defined in <strong>Settings</strong>.
            </p>
            <button className="btn btn-secondary btn-sm" style={{ marginTop: 12, width: '100%' }}>Configure Schedule</button>
          </div>

          <div className="card glass">
            <h3 style={{ fontSize: '1rem', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Filter size={16} className="v-helmet" /> Format Information
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Reports are exported in standard <strong>CSV</strong> (Comma Separated Values) format for direct compatibility with Microsoft Excel, Google Sheets, and other BI tools like Tableau or PowerBI.
            </p>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 12, padding: '8px', background: 'rgba(0,0,0,0.2)', borderRadius: 4, fontFamily: 'var(--font-mono)' }}>
              encoding: UTF-8<br/>
              delimiter: ,<br/>
              newline: \n
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
